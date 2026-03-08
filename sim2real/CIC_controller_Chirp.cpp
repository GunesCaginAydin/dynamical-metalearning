// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#if __has_include(<filesystem>)
  #include <filesystem>
#endif

#include <Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include "examples_common.h"

struct LogEntry {
  std::array<double, 7> torque;
  Eigen::Vector3d ee_position;
  Eigen::Quaterniond ee_orientation;
  std::array<double, 7> joint_positions;

  double t;           // global time since start of control [s]
  int phase;          // 0 = initial shift, 1 = chirp segment, 2 = between-segment transition (excluded from files)
  int circle_idx;     // -1 during shift, 0..N-1 during segments, -2 during transitions
  double t_in_circle; // [s], valid when phase==1 (local time in segment)
};

static void save_csv(const std::string& path, const std::vector<LogEntry>& log, bool include_phase_cols=true) {
  std::ofstream file(path);
  if (!file.is_open()) throw std::runtime_error("Failed to open log file: " + path);

  if (include_phase_cols) {
    file << "t,phase,circle_idx,t_in_circle,";
  } else {
    file << "t,";
  }
  file << "tau0,tau1,tau2,tau3,tau4,tau5,tau6,"
       << "px,py,pz,"
       << "qx,qy,qz,qw,"
       << "q0,q1,q2,q3,q4,q5,q6\n";

  for (const auto& e : log) {
    if (include_phase_cols) {
      file << e.t << "," << e.phase << "," << e.circle_idx << "," << e.t_in_circle << ",";
    } else {
      file << e.t << ",";
    }

    for (double tau : e.torque) file << tau << ",";
    file << e.ee_position(0) << "," << e.ee_position(1) << "," << e.ee_position(2) << ","
         << e.ee_orientation.x() << "," << e.ee_orientation.y() << ","
         << e.ee_orientation.z() << "," << e.ee_orientation.w() << ",";
    for (double q : e.joint_positions) file << q << ",";
    file.seekp(-1, std::ios_base::cur);
    file << "\n";
  }
  file.close();
}

static inline double smoothstep_cos(double u) {
  return 0.5 * (1.0 - std::cos(M_PI * u));
}

// Linear chirp phase in radians, using Hz endpoints over duration T.
// theta(t) = 2π ( f0 t + 0.5 * (f1-f0)/T * t^2 ) + phi
static inline double chirp_phase(double t, double T, double f0_hz, double f1_hz, double phi) {
  const double k = (f1_hz - f0_hz) / std::max(1e-9, T);
  return 2.0 * M_PI * (f0_hz * t + 0.5 * k * t * t) + phi;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  // Compliance parameters
  const double translational_stiffness{300.0};
  const double rotational_stiffness{50.0};
  Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
  stiffness.setZero();
  damping.setZero();
  stiffness.topLeftCorner(3, 3) = translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) = rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  damping.topLeftCorner(3, 3) = 2.0 * std::sqrt(translational_stiffness) * Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) = 2.0 * std::sqrt(rotational_stiffness) * Eigen::MatrixXd::Identity(3, 3);

  // ----------------- TRAJECTORY PARAMETERS (BASE) -----------------
  // "radius_base": typical XY excursion amplitude.
  const double radius_base = 0.10;   // m
  const int num_circles = 5;         // number of segments

  // amplitude variations
  const double radius_jitter = 0.05; // +/- m

  // Chirp frequency bounds (Hz) and jitter
  const double f0_base = 0.10;       // Hz start
  const double f1_base = 0.60;       // Hz end
  const double f_jitter = 0.05;      // +/- Hz

  // Segment duration
  const double T_base = 6.0;         // s per segment
  const double T_jitter = 1.5;       // +/- s (optional)

  // Z variation per segment: +/- 10 cm (constant within segment)
  const double z_jitter = 0.10;      // m

  // Between-segments transition (excluded from saved files)
  const double transition_duration = 2.0; // s
  // ---------------------------------------------------------------

  // Shift parameters (initial shift only)
  const double x_offset = 0.15;     // m
  const double move_duration = 4.0; // s

  try {
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);
    franka::Model model = robot.loadModel();

    franka::RobotState initial_state = robot.readOnce();

    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
    Eigen::Vector3d position_d(initial_transform.translation());
    Eigen::Quaterniond orientation_d(initial_transform.rotation());

    const Eigen::Vector3d center = position_d;
    Eigen::Vector3d shifted_center = center;
    shifted_center(0) += x_offset;

    robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

#if __has_include(<filesystem>)
    std::filesystem::create_directories("data");
#endif

    std::cout << "WARNING: Collision thresholds are set to high values. "
              << "Make sure you have the user stop at hand!\n"
              << "Press Enter to continue...\n";
    std::cin.ignore();

    // ----------------- PRECOMPUTE PER-SEGMENT CHIRP PARAMS -----------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ur(-1.0, 1.0);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    const int num_trajs = num_circles;

    std::vector<double> Ax(num_trajs), Ay(num_trajs);
    std::vector<double> f0x(num_trajs), f1x(num_trajs), f0y(num_trajs), f1y(num_trajs);
    std::vector<double> phx(num_trajs), phy(num_trajs);
    std::vector<double> T_i(num_trajs), z_i(num_trajs);

    auto clamp_pos = [](double v, double lo, double hi) {
      return std::max(lo, std::min(hi, v));
    };

    for (int i = 0; i < num_trajs; ++i) {
      // Amplitudes
      const double A = clamp_pos(radius_base + radius_jitter * ur(rng), 0.01, 0.25);
      // allow slightly different amplitude in x/y
      Ax[i] = clamp_pos(A * (0.7 + 0.6 * u01(rng)), 0.01, 0.30);
      Ay[i] = clamp_pos(A * (0.7 + 0.6 * u01(rng)), 0.01, 0.30);

      // Duration
      T_i[i] = clamp_pos(T_base + T_jitter * ur(rng), 2.0, 12.0);

      // Z constant in segment
      z_i[i] = shifted_center(2) + z_jitter * ur(rng);

      // Frequencies (Hz) with jitter
      const double f0 = clamp_pos(f0_base + f_jitter * ur(rng), 0.02, 1.5);
      const double f1 = clamp_pos(f1_base + f_jitter * ur(rng), 0.05, 2.5);

      // Option A: sweep up (f0 -> f1) always
      f0x[i] = f0; f1x[i] = f1;
      f0y[i] = f0; f1y[i] = f1;

      // Random phase offsets
      phx[i] = 2.0 * M_PI * u01(rng);
      phy[i] = 2.0 * M_PI * u01(rng);
    }

    auto eval_traj = [&](int i, double tau) -> Eigen::Vector3d {
      const double T = T_i[i];
      const double thx = chirp_phase(tau, T, f0x[i], f1x[i], phx[i]);
      const double thy = chirp_phase(tau, T, f0y[i], f1y[i], phy[i]);

      const double x = shifted_center(0) + Ax[i] * std::sin(thx);
      const double y = shifted_center(1) + Ay[i] * std::sin(thy);
      const double z = z_i[i];
      return Eigen::Vector3d(x, y, z);
    };

    // ----------------- BUILD TIMELINE (segments + transitions) -----------------
    struct Segment {
      int type;       // 1=traj segment, 2=transition
      int idx;        // segment index for type 1; transition idx refers to "between idx and idx+1"
      double t_start; // start time since segments started (tc)
      double t_end;   // end time since segments started (tc)
    };
    std::vector<Segment> segments;
    segments.reserve(num_trajs * 2);

    double acc = 0.0;
    for (int i = 0; i < num_trajs; ++i) {
      segments.push_back(Segment{1, i, acc, acc + T_i[i]});
      acc += T_i[i];
      if (i != num_trajs - 1) {
        segments.push_back(Segment{2, i, acc, acc + transition_duration});
        acc += transition_duration;
      }
    }
    const double segments_total_time = acc;
    const double total_time = move_duration + segments_total_time;

    // ------------ LOG BUFFER (in-memory) ------------
    std::vector<LogEntry> log;
    log.reserve(static_cast<size_t>(total_time * 1200.0));

    double t = 0.0;

    auto cb = [&](const franka::RobotState& robot_state,
                  franka::Duration dt) -> franka::Torques {
      t += dt.toSec();

      // dynamics
      std::array<double, 7> coriolis_array = model.coriolis(robot_state);
      std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

      Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      Eigen::Vector3d position(transform.translation());
      Eigen::Quaterniond orientation(transform.rotation());

      // phase bookkeeping
      int phase = 0;
      int circle_idx = -1;
      double t_in_circle = 0.0;

      if (t < move_duration) {
        // Phase 0: initial smooth X shift only
        const double u = std::min(1.0, t / move_duration);
        const double s = smoothstep_cos(u);
        position_d = center;
        position_d(0) = center(0) + s * x_offset;
      } else {
        const double tc = t - move_duration;

        if (tc >= segments_total_time) {
          const int last = num_trajs - 1;
          phase = 1;
          circle_idx = last;
          t_in_circle = T_i[last];
          position_d = eval_traj(last, t_in_circle);
        } else {
          const Segment* seg = nullptr;
          for (const auto& s : segments) {
            if (tc >= s.t_start && tc < s.t_end) { seg = &s; break; }
          }

          if (seg && seg->type == 1) {
            phase = 1;
            circle_idx = seg->idx;
            t_in_circle = tc - seg->t_start;
            position_d = eval_traj(circle_idx, t_in_circle);

          } else if (seg && seg->type == 2) {
            phase = 2;
            circle_idx = -2;
            t_in_circle = 0.0;

            const int i = seg->idx;
            const int j = i + 1;

            const double u = (tc - seg->t_start) / (seg->t_end - seg->t_start);
            const double s = smoothstep_cos(std::min(1.0, std::max(0.0, u)));

            const Eigen::Vector3d p0 = eval_traj(i, T_i[i]);
            const Eigen::Vector3d p1 = eval_traj(j, 0.0);
            position_d = p0 + s * (p1 - p0);
          }
        }
      }

      // impedance errors
      Eigen::Matrix<double, 6, 1> error;
      error.head(3) = position - position_d;

      // orientation error (keep initial orientation)
      if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation.coeffs() = -orientation.coeffs();
      }
      Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
      error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
      error.tail(3) = -transform.rotation() * error.tail(3);

      // control
      Eigen::VectorXd tau_task(7), tau_d(7);
      tau_task = jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
      tau_d = tau_task + coriolis;

      std::array<double, 7> tau_d_array{};
      Eigen::VectorXd::Map(tau_d_array.data(), 7) = tau_d;

      // LOG (every control step)
      log.push_back(LogEntry{tau_d_array, position, orientation, robot_state.q, t, phase, circle_idx, t_in_circle});

      if (t >= total_time) {
        return franka::MotionFinished(franka::Torques(tau_d_array));
      }

      return tau_d_array;
    };

    robot.control(cb);

    // ---------------- SAVE AFTER CONTROL ----------------
    {
      std::vector<LogEntry> shift_log;
      shift_log.reserve(10000);
      for (const auto& e : log) {
        if (e.phase == 0) shift_log.push_back(e);
      }
      const std::string shift_file = "data/log_shift.csv";
      save_csv(shift_file, shift_log, true);
      std::cout << "Saved: " << shift_file << " (" << shift_log.size() << " samples)\n";
    }

    for (int c = 0; c < num_trajs; ++c) {
      std::vector<LogEntry> seg_log;
      seg_log.reserve(10000);
      for (const auto& e : log) {
        if (e.phase == 1 && e.circle_idx == c) seg_log.push_back(e);
      }

      std::ostringstream fname;
      fname << "data/log_chirp_" << std::setw(3) << std::setfill('0') << c
            << "_Ax" << std::fixed << std::setprecision(4) << Ax[c]
            << "_Ay" << std::fixed << std::setprecision(4) << Ay[c]
            << "_f0" << std::fixed << std::setprecision(4) << f0x[c]
            << "_f1" << std::fixed << std::setprecision(4) << f1x[c]
            << "_T"  << std::fixed << std::setprecision(4) << T_i[c]
            << "_z"  << std::fixed << std::setprecision(4) << z_i[c]
            << ".csv";

      save_csv(fname.str(), seg_log, true);
      std::cout << "Saved: " << fname.str() << " (" << seg_log.size() << " samples)\n";
    }

    std::cout << "Done. Total samples: " << log.size() << "\n";

  } catch (const franka::Exception& ex) {
    std::cout << ex.what() << std::endl;
    return -1;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return -1;
  }

  return 0;
}
