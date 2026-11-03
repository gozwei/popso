#pragma once
// A small fixed pool of worker threads for data-parallel loops (used by the barrier's linear
// algebra). run(tasks, fn) calls fn(0) .. fn(tasks - 1) on the workers and the calling thread
// and returns when all calls are done and every worker has finished with this round (so no worker
// can see a later round's data). Not reentrant: one run() at a time per pool.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace milp {

class WorkerPool {
 public:
  explicit WorkerPool(int threads) {
    for (int t = 1; t < threads; ++t) workers_.emplace_back([this] { loop(); });
  }
  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      quit_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  void run(int tasks, const std::function<void(int)>& fn) {
    if (tasks <= 0) return;
    if (workers_.empty() || tasks == 1) {
      for (int t = 0; t < tasks; ++t) fn(t);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = &fn;
      tasks_ = tasks;
      next_.store(0);
      done_ = 0;
      acked_ = 0;
      ++generation_;
    }
    cv_.notify_all();
    const int mine = work(&fn, tasks);
    std::unique_lock<std::mutex> lk(mu_);
    done_ += mine;
    done_cv_.wait(lk, [&] { return done_ == tasks_ && acked_ == static_cast<int>(workers_.size()); });
    fn_ = nullptr;
  }

 private:
  // fn and tasks are read under the lock by the caller of work().
  int work(const std::function<void(int)>* fn, int tasks) {
    int finished = 0;
    for (;;) {
      const int t = next_.fetch_add(1);
      if (t >= tasks) break;
      (*fn)(t);
      ++finished;
    }
    return finished;
  }
  void loop() {
    unsigned long long seen = 0;
    for (;;) {
      const std::function<void(int)>* fn;
      int tasks;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return quit_ || generation_ != seen; });
        if (quit_) return;
        seen = generation_;
        fn = fn_;
        tasks = tasks_;
      }
      const int finished = work(fn, tasks);
      std::lock_guard<std::mutex> lk(mu_);
      done_ += finished;
      ++acked_;
      if (acked_ == static_cast<int>(workers_.size())) done_cv_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int)>* fn_ = nullptr;
  int tasks_ = 0;
  std::atomic<int> next_{0};
  int done_ = 0;
  int acked_ = 0;
  unsigned long long generation_ = 0;
  bool quit_ = false;
};

}  // namespace milp
