// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

ThreadPool::ThreadPool(unsigned int n) : busy(), processed(), stop() {
  for (unsigned int i = 0; i < n; ++i)
    workers.emplace_back(std::bind(&ThreadPool::thread_proc, this));
}

ThreadPool::~ThreadPool() {
  // set stop-condition
  std::unique_lock<std::mutex> latch(queue_mutex);
  stop = true;
  cv_task.notify_all();
  latch.unlock();

  // all threads terminate, then we're done.
  for (auto &t : workers)
    t.join();
}

void ThreadPool::thread_proc() {
  while (true) {
    std::unique_lock<std::mutex> latch(queue_mutex);
    cv_task.wait(latch, [this]() { return stop || !tasks.empty(); });
    if (!tasks.empty()) {
      // got work. set busy.
      ++busy;

      // pull from queue
      auto fn = tasks.front();
      tasks.pop_front();

      // release lock. run async
      latch.unlock();

      // run function outside context
      fn();
      ++processed;

      latch.lock();
      --busy;
      cv_finished.notify_one();
    } else if (stop) {
      break;
    }
  }
}

// generic function push
template <class F> void ThreadPool::enqueue(F &&f) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  tasks.emplace_back(std::forward<F>(f));
  cv_task.notify_one();
}

// waits until the queue is empty.
void ThreadPool::waitFinished() {
  std::unique_lock<std::mutex> lock(queue_mutex);
  cv_finished.wait(lock, [this]() { return tasks.empty() && (busy == 0); });
}

const int MAX_CONCURRENCY = std::thread::hardware_concurrency();
ThreadPool pool(MAX_CONCURRENCY);

std::vector<std::pair<int, int>> split_work(int num) {
  std::vector<int> work;
  if (num < MAX_CONCURRENCY) {
    work = std::vector<int>(num, 1);
  } else {
    work = std::vector<int>(MAX_CONCURRENCY, num / MAX_CONCURRENCY);
    for (int i = 0; i < num % MAX_CONCURRENCY; ++i) {
      work[i] += 1;
    }
  }
  std::vector<std::pair<int, int>> ret(work.size());
  int cum_sum = 0;
  for (std::size_t i = 0; i < work.size(); ++i) {
    ret[i] = std::make_pair(cum_sum, work[i]);
    cum_sum += work[i];
  }
  return ret;
}

struct DPNode;

struct Node {
  int id;
  int father_id;

  int cut_len;
  std::vector<Node *> cut_nodes;

  int p_num;
  std::vector<double> p_time;
  std::vector<int> p_comp_mem;
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
  int pg_id;
  std::vector<int> ir;
  std::vector<std::pair<DPNode *, double>> in_edges;
  // mem, time, activation_mem, optimzer_mem
  std::vector<std::tuple<int, double, DPNode *, int, int>> state;
};

void resetNode(Node *node) {
  for (DPNode *dp_node : node->dp_nodes) {
    dp_node->state.clear();
  }
}

void printNode(Node *node) {
  std::cout << "id: " << node->id << std::endl;
  std::cout << "father_id: " << node->father_id << std::endl;
  std::cout << "cut_len: " << node->cut_len << std::endl;
  std::cout << "cut_nodes: ";
  for (auto cut_node : node->cut_nodes) {
    std::cout << cut_node->id << " ";
  }
  std::cout << std::endl;
  std::cout << "p_num: " << node->p_num << std::endl;
  std::cout << "p_time: ";
  for (auto p_time : node->p_time) {
    std::cout << p_time << " ";
  }
  std::cout << std::endl;
  std::cout << "p_comp_mem: ";
  for (auto p_comp_mem : node->p_comp_mem) {
    std::cout << p_comp_mem << " ";
  }
  std::cout << std::endl;
  std::cout << "p_buf_mem: ";
  for (auto p_buf_mem : node->p_buf_mem) {
    std::cout << p_buf_mem << " ";
  }
  std::cout << std::endl;
  std::cout << "p_act_mem: ";
  for (auto p_act_mem : node->p_act_mem) {
    std::cout << p_act_mem << " ";
  }
  std::cout << std::endl;
  std::cout << "p_opt_mem: ";
  for (auto p_opt_mem : node->p_opt_mem) {
    std::cout << p_opt_mem << " ";
  }
  std::cout << std::endl;
  std::cout << "producer_num: " << node->producer_num << std::endl;
  std::cout << "producers: ";
  for (auto producer : node->producers) {
    std::cout << producer->id << " ";
  }
  std::cout << std::endl;
  std::cout << "p_father: ";
  for (auto p_father : node->p_father) {
    std::cout << p_father << " ";
  }
  std::cout << std::endl;
  std::cout << "comm_costs: " << std::endl;
  for (auto comm_cost : node->comm_costs) {
    for (auto cost : comm_cost) {
      std::cout << cost << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "dp_num: " << node->dp_num << std::endl;
  std::cout << std::endl;
}

// lazy decode
// after decoding, ir stores the partition id of each cut node
void decodePGID(DPNode *dp_node) {
  if (!dp_node->ir.empty()) {
    return;
  }
  Node *node = dp_node->graph_node;
  int val = dp_node->pg_id;
  for (int i = 0; i < node->cut_len; ++i) {
    Node *cur_node = node->cut_nodes[node->cut_len - i - 1];
    dp_node->ir.push_back(val % cur_node->p_num);
    val /= cur_node->p_num;
  }
  std::reverse(dp_node->ir.begin(), dp_node->ir.end());
}

struct SearchPlan {
  double all_time;
  double inner_time;
  int memory;
  std::vector<std::pair<int, int>> path;

  bool operator<(const SearchPlan &other) const {
    return all_time < other.all_time;
  }
};

class DPSolver {
public:
  DPSolver(bool verbose, int mode, int mem_bound, int mem_div, int topk) : verbose(verbose), mode(mode), mem_bound(mem_bound), mem_div(mem_div), topk(topk)
                {
    queries.clear();
    id2node.clear();
    search_results.clear();
  }

  void add_interval(int start, int end) {
    if (verbose) {
      std::cout << "add interval start: " << start << ", end: " << end
                << std::endl;
    }
    queries.push_back(std::make_pair(start, end));
  }

  void add_node(int id, int father_id, std::vector<int> cut_ids,
                std::vector<int> producers, int p_num) {
    if (verbose) {
      std::cout << "id: " << id << ", father_id: " << father_id
                << ", cut_ids: ";
      for (int cut_id : cut_ids) {
        std::cout << cut_id << " ";
      }
      std::cout << ", producers: ";
      for (int producer : producers) {
        std::cout << producer << " ";
      }
      std::cout << ", p_num: " << p_num << std::endl;
    }
    Node *node = new Node();
    id2node[id] = node;
    node->id = id;
    node->p_num = p_num;
    node->p_father.resize(p_num);
    node->p_time.resize(p_num);
    node->p_comp_mem.resize(p_num);
    node->p_buf_mem.resize(p_num);
    node->p_act_mem.resize(p_num);
    node->p_opt_mem.resize(p_num);
    node->father_id = father_id;
    node->cut_len = cut_ids.size();
    node->cut_nodes.resize(node->cut_len);
    for (int i = 0; i < node->cut_len; ++i) {
      node->cut_nodes[i] = id2node[cut_ids[i]];
    }
    node->producer_num = producers.size();
    node->producers.resize(node->producer_num);
    node->comm_costs.clear();
    node->comm_costs.resize(node->producer_num);
    for (int i = 0; i < node->producer_num; ++i) {
      node->producers[i] = id2node[producers[i]];
      node->comm_costs[i].resize(node->p_num * node->producers[i]->p_num);
    }
  }

  void add_partition(int node_id, int p_idx, double p_time, int p_comp_mem,
                     int p_buf_mem, int p_act_mem, int p_opt_mem, int p_father,
                     std::vector<std::vector<double>> comm_costs) {
    if (verbose) {
      std::cout << "node_id: " << node_id << ", p_idx: " << p_idx
                << ", p_time: " << p_time << ", p_comp_mem: " << p_comp_mem
                << ", p_buf_mem: " << p_buf_mem << ", p_act_mem: " << p_act_mem
                << ", p_opt_mem: " << p_opt_mem << ", p_father: " << p_father
                << std::endl;
      std::cout << "comm_costs: " << std::endl;
      for (std::size_t i = 0; i < comm_costs.size(); ++i) {
        for (std::size_t j = 0; j < comm_costs[i].size(); ++j) {
          std::cout << comm_costs[i][j] << " ";
        }
        std::cout << std::endl;
      }
    }
    Node *node = id2node[node_id];
    node->p_time[p_idx] = p_time;
    node->p_comp_mem[p_idx] = p_comp_mem;
    node->p_buf_mem[p_idx] = p_buf_mem;
    node->p_act_mem[p_idx] = p_act_mem;
    node->p_opt_mem[p_idx] = p_opt_mem;
    node->p_father[p_idx] = p_father;
    for (int i = 0; i < node->producer_num; ++i) {
      for (int j = 0; j < node->producers[i]->p_num; ++j) {
        node->comm_costs[i][p_idx * node->producers[i]->p_num + j] =
            comm_costs[i][j];
      }
    }
  }

  void init_dp_info() {
    for (auto iter = id2node.begin(); iter != id2node.end(); ++iter) {
      Node *node = iter->second;
      node->dp_num = 1;
      for (Node *cut_node : node->cut_nodes) {
        node->dp_num *= cut_node->p_num;
      }
      node->dp_nodes.resize(node->dp_num);
      // pg: partition group, denotes the maintained partition states in
      // a node. to reduce memory usage, we use a single int to
      // represent a partition group
      for (int j = 0; j < node->dp_num; ++j) {
        DPNode *dp_node = new DPNode();
        node->dp_nodes[j] = dp_node;
        dp_node->graph_node = node;
        dp_node->pg_id = j;
        dp_node->ir.clear();
        dp_node->in_edges.clear();
        dp_node->state.clear();
      }
    }
  }

  // lazy build edge
  void buildInEdges(DPNode *dp_node) {
    if (!dp_node->in_edges.empty()) {
      return;
    }
    Node *node = dp_node->graph_node;

    // special case: the node does not have any producer
    // the pred dp node is composed of the same cut nodes as the current node
    // except the last one. the transition cost is 0 since there is no
    // communication
    if (node->producer_num == 0) {
      int val = 0;
      for (int i = 0; i < node->cut_len - 1; ++i) {
        val += dp_node->ir[i];
        if (i < node->cut_len - 2) {
          val *= node->cut_nodes[i + 1]->p_num;
        }
      }
      Node *pre_node = id2node[node->id - 1];
      dp_node->in_edges.push_back(std::make_pair(pre_node->dp_nodes[val], 0));
      return;
    }

    int cur_p = *(dp_node->ir.rbegin());
    // we have filtered out the partition that cannot find a father to follow
    assert(node->p_father[cur_p] != -1);
    std::map<int, int> info;
    for (int i = 0; i < node->cut_len - 1; ++i) {
      info[node->cut_nodes[i]->id] = dp_node->ir[i];
    }
    // TODO(yizhu1): optimize
    int producer_comb_num = 1;
    for (Node *producer : node->producers) {
      producer_comb_num *= producer->p_num;
    }
    // enumerate all the possible producer partition combinations
    // to build the in edges
    for (int idx = 0; idx < producer_comb_num; ++idx) {
      bool is_legal = true;
      int val = idx;
      std::vector<int> producer_ps(node->producer_num);
      // decode the producer partition combination
      for (int j = 0; j < node->producer_num; ++j) {
        int k = node->producer_num - 1 - j;
        producer_ps[k] = val % node->producers[k]->p_num;
        val /= node->producers[k]->p_num;
        // constraint: if the producer shares the same father with the node,
        // then the partition of the node should follow the producer's
        // partition, except the producer is the father node.
        if (node->father_id != node->id) {
          Node *producer = node->producers[k];
          // TODO: do we need to check producer->father_id != producer->id?
          // seems this case will be filtered out by checker in line 411
          if (producer->father_id == node->father_id &&
              producer->father_id != producer->id) {
            if (node->p_father[cur_p] != producer->p_father[producer_ps[k]]) {
              is_legal = false;
            }
          }
        }
      }
      if (!is_legal) {
        continue;
      }
      // <cut_node_id, cut_node_partition_id>
      std::vector<std::pair<int, int>> cur_ir(node->cut_len - 1);
      bool has_found_follow = false;
      for (int i = 0; i < node->cut_len - 1; ++i) {
        cur_ir[i] = std::make_pair(node->cut_nodes[i]->id, dp_node->ir[i]);
        if (node->cut_nodes[i]->father_id == node->father_id) {
          has_found_follow = true;
        }
      }
      double cost = 0;
      std::vector<std::pair<int, int>> follow_candidates;
      for (int j = 0; j < node->producer_num; ++j) {
        int producer_id = node->producers[j]->id;
        int producer_p = producer_ps[j];
        auto iter = info.find(producer_id);
        if (iter != info.end()) {
          if (producer_p != iter->second) {
            is_legal = false;
            break;
          }
        } else {
          Node *producer = node->producers[j];
          if (producer->father_id != node->father_id) {
            // check that there is a existing node in cur_ir that in the same
            // follow chain with the producer
            bool find_existing_follow = false;
            for (std::size_t i = 0; i < cur_ir.size(); ++i) {
              Node *tmp = id2node[cur_ir[i].first];
              if (tmp->father_id == producer->father_id) {
                find_existing_follow = true;
                // update
                if (tmp->id < producer->id) {
                  for (int _ = 0; _ < producer->p_num; ++_) {
                    if (producer->p_father[_] ==
                        tmp->p_father[cur_ir[i].second]) {
                      // replace to align with the filter logic in python
                      // only the newest node in the follow chain is kept
                      cur_ir[i] = std::make_pair(producer->id, _);
                      break;
                    }
                  }
                }
                break;
              }
            }
            if (!find_existing_follow) {
              cur_ir.push_back(std::make_pair(producer_id, producer_p));
            }
          } else {
            follow_candidates.push_back(
                std::make_pair(producer_id, producer_p));
          }
        }
        cost +=
            node->comm_costs[j][cur_p * node->producers[j]->p_num + producer_p];
      }
      if (!is_legal) {
        continue;
      }
      // handle follow
      bool find_pre_id = false;
      for (std::size_t j = 0; j < cur_ir.size(); ++j) {
        if (cur_ir[j].first == node->id - 1) {
          find_pre_id = true;
          break;
        }
      }
      if (!find_pre_id) {
        Node *pre_node = id2node[node->id - 1];
        if (pre_node->father_id != node->father_id) {
          // do nothing, means the pre_node's output is not used
          // we select the 1st partition of the pre_node
          // need to be careful when the graph has multiple outputs
          // shall we constrain that the output of the graph is replicated?
        } else if (pre_node->father_id == pre_node->id) {
          assert(follow_candidates.rbegin()->first == pre_node->id);
          cur_ir.push_back(*follow_candidates.rbegin());
        } else {
          bool find_same_follow_p = false;
          for (int k = 0; k < pre_node->p_num; ++k) {
            if (pre_node->p_father[k] == node->p_father[cur_p]) {
              cur_ir.push_back(std::make_pair(node->id - 1, k));
              find_same_follow_p = true;
              break;
            }
          }
          assert(find_same_follow_p);
        }
      } else {
        if (node->father_id != node->id && !has_found_follow &&
            !follow_candidates.empty()) {
          cur_ir.push_back(*follow_candidates.rbegin());
        }
      }
      std::sort(cur_ir.begin(), cur_ir.end());
      val = 0;
      for (std::size_t j = 0; j < cur_ir.size(); ++j) {
        val += cur_ir[j].second;
        if (j + 1 < cur_ir.size()) {
          val *= id2node[cur_ir[j + 1].first]->p_num;
        }
      }
      dp_node->in_edges.push_back(
          std::make_pair(id2node[node->id - 1]->dp_nodes[val], cost));
    }
  }

  // do dp for a partition group
  void update(DPNode *dp_node, int start_level) {
    Node *node = dp_node->graph_node;
    decodePGID(dp_node);
    int cur_p_idx = *(dp_node->ir.rbegin());
    if (node->id == start_level) {
      // each dp node maintains a list of states, each state is a tuple
      // (mem, time, pred_dp_node, activation_mem, optimizer_mem)
      dp_node->state.push_back(std::make_tuple(
          node->p_comp_mem[cur_p_idx], node->p_time[cur_p_idx], nullptr,
          node->p_act_mem[cur_p_idx], node->p_opt_mem[cur_p_idx]));
      return;
    }

    // storing edges takes space, so we build edges when needed
    buildInEdges(dp_node);
    int cur_p = *(dp_node->ir.rbegin());
    if (dp_node->in_edges.empty()) {
      dp_node->state.push_back(std::make_tuple(
          0, std::numeric_limits<double>::infinity(), nullptr, 0, 0));
      return;
    }

    // use a priority queue to maintain the best state, similar to the merge
    // sort
    double cur_p_time = node->p_time[cur_p];
    int cur_p_comp_mem = node->p_comp_mem[cur_p];
    int cur_p_act_mem = node->p_act_mem[cur_p];
    int cur_p_opt_mem = node->p_opt_mem[cur_p];
    std::priority_queue<std::tuple<int, double, int, int, int>> pq;
    for (std::size_t i = 0; i < dp_node->in_edges.size(); ++i) {
      DPNode *pred = dp_node->in_edges[i].first;
      int mem = cur_p_comp_mem + std::get<0>(pred->state[0]);
      double cost = cur_p_time + dp_node->in_edges[i].second +
                    std::get<1>(pred->state[0]);
      int act_mem = cur_p_act_mem + std::get<3>(pred->state[0]);
      int opt_mem = cur_p_opt_mem + std::get<4>(pred->state[0]);
      pq.push(std::make_tuple(-mem, -cost, i, -act_mem, -opt_mem));
    }

    std::vector<int> lows(dp_node->in_edges.size(), 1);

    int cur_mem;
    double cur_cost;
    int pred_idx;
    int cur_act_mem;
    int cur_opt_mem;
    while (!pq.empty()) {
      std::tie(cur_mem, cur_cost, pred_idx, cur_act_mem, cur_opt_mem) =
          pq.top();
      cur_mem = -cur_mem;
      cur_cost = -cur_cost;
      cur_act_mem = -cur_act_mem;
      cur_opt_mem = -cur_opt_mem;
      pq.pop();
      if (lows[pred_idx] < dp_node->in_edges[pred_idx].first->state.size()) {
        DPNode *pred = dp_node->in_edges[pred_idx].first;
        int mem = cur_p_comp_mem + std::get<0>(pred->state[lows[pred_idx]]);
        double cost = cur_p_time + dp_node->in_edges[pred_idx].second +
                      std::get<1>(pred->state[lows[pred_idx]]);
        int act_mem = cur_p_act_mem + std::get<3>(pred->state[lows[pred_idx]]);
        int opt_mem = cur_p_opt_mem + std::get<4>(pred->state[lows[pred_idx]]);
        pq.push(std::make_tuple(-mem, -cost, pred_idx, -act_mem, -opt_mem));
        ++lows[pred_idx];
      }
      if (dp_node->state.empty()) {
        dp_node->state.push_back(std::make_tuple(
            cur_mem, cur_cost, dp_node->in_edges[pred_idx].first, cur_act_mem,
            cur_opt_mem));
      } else {
        int pre_mem = std::get<0>(dp_node->state[dp_node->state.size() - 1]);
        double pre_cost =
            std::get<1>(dp_node->state[dp_node->state.size() - 1]);
        // if (cur_mem > pre_mem && cur_cost < pre_cost &&
        //     cur_mem + cur_opt_mem <= mem_bound) {
        if (cur_mem > pre_mem && cur_cost < pre_cost &&
            cur_mem - cur_act_mem + std::max(cur_act_mem, cur_opt_mem) <=
                mem_bound) {
          dp_node->state.push_back(std::make_tuple(
              cur_mem, cur_cost, dp_node->in_edges[pred_idx].first, cur_act_mem,
              cur_opt_mem));
        }
      }
    }
  }

  void do_dp(int start_level, int end_level) {
    // reset all the dp nodes, since we may have multiple queries
    for (auto iter = id2node.begin(); iter != id2node.end(); ++iter) {
      resetNode(iter->second);
    }

    for (int i = start_level; i <= end_level; ++i) {
      // use multi-thread to do dp for each level to reduce time
      auto iter = id2node.find(i);
      if (iter == id2node.end()) {
        // TODO(yizhu1): check here
        assert(false);
      }
      if (verbose) {
        std::cout << "Start to process level id: " << i
                  << ", state num: " << iter->second->dp_nodes.size()
                  << std::endl;
      }
      std::vector<std::pair<int, int>> split_info =
          split_work(iter->second->dp_num);
      for (const auto &item : split_info) {
        pool.enqueue([=] {
          for (int i = 0; i < item.second; ++i) {
            int offset = item.first + i;
            update(iter->second->dp_nodes[offset], start_level);
          }
        });
      }
      pool.waitFinished();
    }
  }

  SearchPlan process_state(DPNode *dp_node, int idx) {
    // build the optimal path of each partition of last operator
    // and return the best path
    std::vector<std::pair<int, int>> path;
    DPNode *cur_dp_node = dp_node;
    int cur_idx = idx;
    int best_mem = std::get<0>(dp_node->state[idx]);
    double best_time = std::get<1>(dp_node->state[idx]);
    int act_mem = std::get<3>(dp_node->state[idx]);
    int opt_mem = std::get<4>(dp_node->state[idx]);
    double inner_time = 0;
    int cur_best_mem = best_mem;
    std::vector<int> buffers;
    while (true) {
      int cur_p = *(cur_dp_node->ir.rbegin());
      Node *node = cur_dp_node->graph_node;
      path.push_back(std::make_pair(node->id, cur_p));
      buffers.push_back(node->p_buf_mem[cur_p]);
      inner_time += node->p_time[cur_p];
      cur_best_mem -= node->p_comp_mem[cur_p];
      DPNode *pred_dp_node = std::get<2>(cur_dp_node->state[cur_idx]);
      if (pred_dp_node == nullptr) {
        break;
      } else {
        cur_dp_node = pred_dp_node;
        cur_idx = std::lower_bound(
                      cur_dp_node->state.begin(), cur_dp_node->state.end(),
                      std::make_tuple(cur_best_mem, static_cast<double>(-1),
                                      static_cast<DPNode *>(nullptr), -1, -1)) -
                  cur_dp_node->state.begin();
      }
    }
    std::reverse(path.begin(), path.end());
    std::sort(buffers.begin(), buffers.end());
    long long ret_mem = static_cast<long long>(best_mem);
    if (mode == 0) {
      ret_mem += buffers[buffers.size() - 1] + buffers[buffers.size() - 2];
    } else if (mode == 1) {
      ret_mem += buffers[buffers.size() - 1];
    }
    ret_mem = ret_mem - act_mem + std::max(act_mem, opt_mem);
    if (ret_mem > mem_bound) {
      return SearchPlan{-1, -1, -1, std::vector<std::pair<int, int>>()};
    }
    if (verbose) {
      std::cout << "best time: " << best_time
                << ", best mem: " << best_mem / 1024 / 1024 * mem_div << "MB, "
                << "activation mem: " << act_mem / 1024 / 1024 * mem_div
                << "MB, "
                << "optimizer state mem: " << opt_mem / 1024 / 1024 * mem_div
                << "MB" << std::endl;
    }
    return SearchPlan{best_time, inner_time, static_cast<int>(ret_mem), path};
  }

  void post_process(int start_level, int end_level, int topk) {
    std::vector<SearchPlan> best_info;
    double best_time;
    double inner_time;
    int best_mem;
    std::vector<std::pair<int, int>> path;
    for (DPNode *dp_node : id2node[end_level]->dp_nodes) {
      int cnt = 0;
      for (std::size_t i = 0; i < dp_node->state.size(); ++i) {
        SearchPlan plan = process_state(dp_node, dp_node->state.size() - i - 1);
        if (plan.all_time > 0) {
          ++cnt;
          best_info.push_back(plan);
          if (cnt == topk) {
            break;
          }
        }
      }
    }
    std::sort(best_info.begin(), best_info.end());
    search_results[std::make_pair(start_level, end_level)] = best_info;
  }

  void solve() {
    if (verbose) {
      std::cout << "start to solve" << std::endl;
      std::cout << "verbose: " << verbose << std::endl;
      std::cout << "mode: " << mode << std::endl;
      std::cout << "mem_bound: " << mem_bound << std::endl;
      std::cout << "mem_div: " << mem_div << std::endl;
      std::cout << "topk: " << topk << std::endl;
    }
    init_dp_info();
    // to reduce time, we first group the queries by start node (level)
    std::unordered_map<int, std::vector<int>> intervals;
    for (const auto &query : queries) {
      auto iter = intervals.find(query.first);
      if (iter == intervals.end()) {
        intervals[query.first] = std::vector<int>(1, query.second);
      } else {
        iter->second.push_back(query.second);
      }
    }

    auto start = std::chrono::system_clock::now();
    for (auto &item : intervals) {
      // for each start node, we do dp until the last end node
      int start_level = item.first;
      std::vector<int> &end_levels = item.second;
      std::sort(end_levels.begin(), end_levels.end());
      do_dp(start_level, *end_levels.rbegin());
      for (int end_level : end_levels) {
        post_process(start_level, end_level, topk);
      }
      long long state_cnt = 0;
      for (auto iter = id2node.begin(); iter != id2node.end(); ++iter) {
        int cur_id = iter->first;
        Node *cur_node = iter->second;
        for (DPNode *dp_node : cur_node->dp_nodes) {
          state_cnt += dp_node->state.size();
        }
      }
      if (verbose) {
        std::cout << "state num: " << state_cnt << std::endl;
      }
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << " s"
              << std::endl;
  }

  std::vector<SearchPlan> get_results(int start_level, int end_level) {
    return search_results[std::make_pair(start_level, end_level)];
  }

  bool verbose;
  // mode = 0: training, use the sum of the two largest buffer sizes
  // mode = 1: inference, use the largest buffer size
  int mode;
  // mem_bound: the maximum memory usage, in bytes
  int mem_bound;
  // mem_div: the memory divisor, to avoid overflow in int32
  int mem_div;
  int topk;

  std::unordered_map<int, Node *> id2node;
  std::vector<std::pair<int, int>> queries;
  std::map<std::pair<int, int>, std::vector<SearchPlan>> search_results;
};

PYBIND11_MODULE(dp_solver, m) {
  py::class_<SearchPlan>(m, "SearchPlan")
      .def_readonly("all_time", &SearchPlan::all_time)
      .def_readonly("inner_time", &SearchPlan::inner_time)
      .def_readonly("memory", &SearchPlan::memory)
      .def_readonly("path", &SearchPlan::path);

  py::class_<DPSolver>(m, "DPSolver")
      .def(py::init<bool, int, int, int, int>())
      .def("add_interval", &DPSolver::add_interval)
      .def("add_node", &DPSolver::add_node)
      .def("add_partition", &DPSolver::add_partition)
      .def("solve", &DPSolver::solve)
      .def("get_results", &DPSolver::get_results);
}
/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-std=c++11']
cfg['extra_compile_args'] = ['-O3']
cfg['extra_compile_args'] = ['-pthread']
%>
*/
