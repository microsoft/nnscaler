/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 */

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

class ThreadPool {
public:
  ThreadPool(unsigned int n = std::thread::hardware_concurrency());

  template <class F> void enqueue(F &&f);
  void waitFinished();
  ~ThreadPool();

  unsigned int getProcessed() const { return processed; }

private:
  std::vector<std::thread> workers;
  std::deque<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable cv_task;
  std::condition_variable cv_finished;
  std::atomic_uint processed;
  unsigned int busy;
  bool stop;

  void thread_proc();
};

struct DPNode;
struct UnitDPState;

struct Node {
  int id;
  int father_id;

  int cut_len;
  std::vector<Node *> cut_nodes;

  // whether the node is in a recompute region
  bool is_recompute;
  // if the node is in a recompute region, whether it accetps input outside
  // of the recompute region
  bool is_recompute_in;
  // if the node is in a recompute region, whether it is the last node in the
  // topological sequence of the recompute region
  bool is_recompute_last;

  int p_num;
  std::vector<double> p_time;
  std::vector<int> p_comp_mem;
  std::vector<int> p_in_mem;
  std::vector<int> p_buf_mem;
  std::vector<int> p_act_mem;
  std::vector<int> p_opt_mem;
  std::vector<int> p_father;

  int producer_num;
  std::vector<Node *> producers;
  std::vector<std::vector<double>> comm_costs;

  // assume the number of combinations is less than 2e9
  int dp_num;
  std::vector<DPNode *> dp_nodes;
};

struct DPNode {
  Node *graph_node;
  // pg_id: partition group id, an equivalent representation of `ir`
  int pg_id;
  std::vector<int> ir;
  std::vector<std::pair<DPNode *, double>> in_edges;
  // saved UnitDPStates are sorted by total mem cost and satisfy that
  // if lhs.total_mem < rhs.total_mem, then lhs.time_cost > rhs.time_cost
  std::vector<UnitDPState> state;
};

struct SearchPlan {
  double all_time;
  int memory;
  std::vector<std::pair<int, int>> path;

  bool operator<(const SearchPlan &other) const {
    return all_time < other.all_time;
  }
};

struct UnitDPState {
  int param_related_mem;
  int activation_mem;
  int opt_transient_mem;
  int largest_transient_mem_1st;
  int largest_transient_mem_2nd;
  int max_recompute_mem;
  int cur_recompute_mem;
  int total_mem;
  double time_cost;

  DPNode *pred_dp_node;
  int pred_idx;

  UnitDPState() {}

  UnitDPState(int param_related_mem, int activation_mem,
              int opt_transient_mem, int largest_transient_mem_1st,
              int largest_transient_mem_2nd, int max_recompute_mem,
              int cur_recompute_mem, double time_cost,
              DPNode *pred_dp_node, int pred_idx)
      : param_related_mem(param_related_mem), activation_mem(activation_mem),
        opt_transient_mem(opt_transient_mem),
        largest_transient_mem_1st(largest_transient_mem_1st),
        largest_transient_mem_2nd(largest_transient_mem_2nd),
        max_recompute_mem(max_recompute_mem),
        cur_recompute_mem(cur_recompute_mem), time_cost(time_cost),
        pred_dp_node(pred_dp_node), pred_idx(pred_idx) {
    total_mem = calc_total_mem();
  }

  UnitDPState(Node *node, int partition_idx, DPNode *pred_dp_node, int pred_idx)
      : pred_dp_node(pred_dp_node), pred_idx(pred_idx) {
    double time_cost = node->p_time[partition_idx];
    int comp_mem = node->p_comp_mem[partition_idx];
    int in_mem = node->p_in_mem[partition_idx];
    int buf_mem = node->p_buf_mem[partition_idx];
    int act_mem = node->p_act_mem[partition_idx];
    int opt_mem = node->p_opt_mem[partition_idx];
    this->param_related_mem = comp_mem - act_mem;
    if (node->is_recompute == true) {
      if (node->is_recompute_in == true) {
        this->activation_mem = in_mem;
      } else {
        this->activation_mem = 0;
      }
      this->cur_recompute_mem = act_mem;
      if (node->is_recompute_last == true) {
        this->max_recompute_mem = act_mem;
        this->cur_recompute_mem = 0;
      } else {
        this->max_recompute_mem = 0;
      }
    } else {
      this->activation_mem = act_mem;
      this->max_recompute_mem = 0;
      this->cur_recompute_mem = 0;
    }
    this->opt_transient_mem = opt_mem;
    this->largest_transient_mem_1st = buf_mem;
    this->largest_transient_mem_2nd = 0;
    this->time_cost = time_cost;
    this->total_mem = calc_total_mem();
  }

  int calc_total_mem() {
    return param_related_mem + std::max(activation_mem, opt_transient_mem) +
           largest_transient_mem_1st + largest_transient_mem_2nd +
           max_recompute_mem;
  }

  UnitDPState generate_new_state(Node *node, int partition_idx,
                                 double transition_cost, DPNode *pred_dp_node,
                                 int pred_idx) {
    double time_cost = node->p_time[partition_idx];
    int comp_mem = node->p_comp_mem[partition_idx];
    int in_mem = node->p_in_mem[partition_idx];
    int buf_mem = node->p_buf_mem[partition_idx];
    int act_mem = node->p_act_mem[partition_idx];
    int opt_mem = node->p_opt_mem[partition_idx];
    int param_related_mem = comp_mem - act_mem + this->param_related_mem;
    int opt_transient_mem = opt_mem + this->opt_transient_mem;
    int largest_transient_mem_1st = this->largest_transient_mem_1st;
    int largest_transient_mem_2nd = this->largest_transient_mem_2nd;
    if (buf_mem > largest_transient_mem_1st) {
      largest_transient_mem_2nd = largest_transient_mem_1st;
      largest_transient_mem_1st = buf_mem;
    } else if (buf_mem > largest_transient_mem_2nd) {
      largest_transient_mem_2nd = buf_mem;
    }
    int max_recompute_mem = this->max_recompute_mem;
    int cur_recompute_mem = this->cur_recompute_mem;
    int activation_mem = act_mem + this->activation_mem;
    if (node->is_recompute == true) {
      if (node->is_recompute_in == true) {
        activation_mem = in_mem + this->activation_mem;
      } else {
        activation_mem = this->activation_mem;
      }
      cur_recompute_mem += act_mem;
      if (node->is_recompute_last == true) {
        max_recompute_mem = std::max(max_recompute_mem, cur_recompute_mem);
        cur_recompute_mem = 0;
      }
    }
    return UnitDPState(param_related_mem, activation_mem, opt_transient_mem,
                       largest_transient_mem_1st, largest_transient_mem_2nd,
                       max_recompute_mem, cur_recompute_mem,
                       this->time_cost + transition_cost + time_cost,
                       pred_dp_node, pred_idx);
  }

  std::string to_string() {
    return "param_related_mem: " + std::to_string(param_related_mem) +
           ", activation_mem: " + std::to_string(activation_mem) +
           ", opt_transient_mem: " + std::to_string(opt_transient_mem) +
           ", largest_transient_mem_1st: " +
           std::to_string(largest_transient_mem_1st) +
           ", largest_transient_mem_2nd: " +
           std::to_string(largest_transient_mem_2nd) +
           ", max_recompute_mem: " + std::to_string(max_recompute_mem) +
           ", cur_recompute_mem: " + std::to_string(cur_recompute_mem) +
           ", total_mem: " + std::to_string(total_mem) +
           ", time_cost: " + std::to_string(time_cost);
  }

  // for priority queue, sort the state by total memory cost from small to large
  // if the total memory cost is the same, sort by time cost from small to large
  bool operator<(const UnitDPState &other) const {
    if (total_mem != other.total_mem) {
      return total_mem > other.total_mem;
    } else {
      return time_cost > other.time_cost;
    }
  }
};