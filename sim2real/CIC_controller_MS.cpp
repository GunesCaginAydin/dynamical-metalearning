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
  int phase;          // 0 = initial shift, 1 = multisin segment, 2 = between-segment transition (excluded from files)
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
  const double radius_base = 0.10;  // m 
  const double omega_base  = 1.20;  // rad/s (fundamental frequency)
  const int num_circles = 5;        // number of segments

  // Small per-segment variations
  const double radius_jitter = 0.05; // +/- m 
  const double omega_jitter  = 0.20; // +/- rad/s

  // Z variation per segment: +/- 10 cm (constant Z per segment)
  const double z_jitter = 0.10; // m

  // Between-segments transition (excluded from saved files)
  const double transition_duration = 2.0; // s

  // Multisin content
  const int K = 4;

  // Shift parameters (initial shift only)
  const double x_offset = 0.15;     // m
  const double move_duration = 4.0; // seconds

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

    // ----------------- PRECOMPUTE PER-SEGMENT MULTISINUSOID PARAMS -----------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ur(-1.0, 1.0);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    const int num_trajs = num_circles;

    std::vector<double> A_xy(num_trajs), w0(num_trajs), T_i(num_trajs), z_i(num_trajs);
    std::vector<std::array<double, K>> ax(num_trajs), ay(num_trajs);
    std::vector<std::array<double, K>> phx(num_trajs), phy(num_trajs);

    auto normalize_l1_to_budget = [&](std::array<double, K>& a, double budget) {
      double s = 0.0;
      for (double v : a) s += std::abs(v);
      if (s < 1e-9) { a.fill(0.0); a[0] = budget; return; }
      const double scale = budget / s;
      for (double& v : a) v *= scale;
    };

    for (int i = 0; i < num_trajs; ++i) {
      A_xy[i] = radius_base + radius_jitter * ur(rng);
      if (A_xy[i] < 0.01) A_xy[i] = 0.01;

      w0[i] = omega_base + omega_jitter * ur(rng);
      if (w0[i] < 0.05) w0[i] = 0.05;

      // Segment repeats over this period (harmonics are integer multiples)
      T_i[i] = 2.0 * M_PI / w0[i];

      z_i[i] = shifted_center(2) + z_jitter * ur(rng);

      // Random raw amplitudes + phases
      for (int k = 0; k < K; ++k) {
        ax[i][k] = ur(rng);
        ay[i][k] = ur(rng);
        phx[i][k] = 2.0 * M_PI * u01(rng);
        phy[i][k] = 2.0 * M_PI * u01(rng);
      }

      const double bx = A_xy[i] * (0.5 + 0.5 * u01(rng));
      const double by = A_xy[i] * (0.5 + 0.5 * u01(rng));

      normalize_l1_to_budget(ax[i], bx);
      normalize_l1_to_budget(ay[i], by);
    }

    auto eval_traj = [&](int i, double tau) -> Eigen::Vector3d {
      double x = shifted_center(0);
      double y = shifted_center(1);
      double z = z_i[i];

      for (int k = 0; k < K; ++k) {
        const double wk = (k + 1) * w0[i];
        x += ax[i][k] * std::sin(wk * tau + phx[i][k]);
        y += ay[i][k] * std::sin(wk * tau + phy[i][k]);
      }
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

      std::array<double, 7> coriolis_array = model.coriolis(robot_state);
      std::array<double, 42> jacobian_array =
          model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
      Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

      Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      Eigen::Vector3d position(transform.translation());
      Eigen::Quaterniond orientation(transform.rotation());

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
            // Phase 1: multisinusoidal trajectory segment
            phase = 1;
            circle_idx = seg->idx;
            t_in_circle = tc - seg->t_start;

            const int i = circle_idx;
            position_d = eval_traj(i, t_in_circle);

          } else if (seg && seg->type == 2) {
            // Phase 2: transition between segments (NOT saved)
            phase = 2;
            circle_idx = -2;
            t_in_circle = 0.0;

            const int i = seg->idx; // transition from segment i -> i+1
            const int j = i + 1;

            const double u = (tc - seg->t_start) / (seg->t_end - seg->t_start); // 0..1
            const double s = smoothstep_cos(std::min(1.0, std::max(0.0, u)));

            const Eigen::Vector3d p0 = eval_traj(i, T_i[i]); // end of segment i
            const Eigen::Vector3d p1 = eval_traj(j, 0.0);    // start of segment j
            position_d = p0 + s * (p1 - p0);

          } else {
            // fallback: hold current desired pose
            position_d = position_d;
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

      // finish condition: initial shift + entire segments timeline
      if (t >= total_time) {
        return franka::MotionFinished(franka::Torques(tau_d_array));
      }

      return tau_d_array;
    };

    robot.control(cb);

    // ---------------- SAVE AFTER CONTROL ----------------
    // Save ONLY initial shift (phase==0). Transitions (phase==2) excluded.
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

    // Save each multisin segment as its own file (phase==1 && circle_idx==c). Transitions excluded.
    for (int c = 0; c < num_trajs; ++c) {
      std::vector<LogEntry> seg_log;
      seg_log.reserve(10000);
      for (const auto& e : log) {
        if (e.phase == 1 && e.circle_idx == c) seg_log.push_back(e);
      }

      std::ostringstream fname;
      fname << "data/log_multisin_" << std::setw(3) << std::setfill('0') << c
            << "_A"  << std::fixed << std::setprecision(4) << A_xy[c]
            << "_w0" << std::fixed << std::setprecision(4) << w0[c]
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
