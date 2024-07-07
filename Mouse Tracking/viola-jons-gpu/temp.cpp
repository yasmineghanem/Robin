// #include <iostream>
// #include <vector>
// #include <thread>
// #include <queue>
// #include <functional>
// #include <mutex>
// #include <condition_variable>
// #include <future>

// class ThreadPool
// {
// public:
//     ThreadPool(size_t threads);
//     ~ThreadPool();

//     template <class F, class... Args>
//     auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

// private:
//     std::vector<std::thread> workers;
//     std::queue<std::function<void()>> tasks;

//     std::mutex queue_mutex;
//     std::condition_variable condition;
//     bool stop;
// };

// // Constructor
// ThreadPool::ThreadPool(size_t threads) : stop(false)
// {
//     for (size_t i = 0; i < threads; ++i)
//         workers.emplace_back(
//             [this]
//             {
//                 for (;;)
//                 {
//                     std::function<void()> task;
//                     {
//                         std::unique_lock<std::mutex> lock(this->queue_mutex);
//                         this->condition.wait(lock, [this]
//                                              { return this->stop || !this->tasks.empty(); });
//                         if (this->stop && this->tasks.empty())
//                             return;
//                         task = std::move(this->tasks.front());
//                         this->tasks.pop();
//                     }
//                     task();
//                 }
//             });
// }

// // Destructor
// ThreadPool::~ThreadPool()
// {
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);
//         stop = true;
//     }
//     condition.notify_all();
//     for (std::thread &worker : workers)
//         worker.join();
// }

// // Add new work item to the pool
// template <class F, class... Args>
// auto ThreadPool::enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
// {
//     using return_type = typename std::result_of<F(Args...)>::type;

//     auto task = std::make_shared<std::packaged_task<return_type()>>(
//         std::bind(std::forward<F>(f), std::forward<Args>(args)...));

//     std::future<return_type> res = task->get_future();
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);
//         if (stop)
//             throw std::runtime_error("enqueue on stopped ThreadPool");
//         tasks.emplace([task]()
//                       { (*task)(); });
//     }
//     condition.notify_one();
//     return res;
// }

// int main()
// {
//     ThreadPool pool(4);

//     auto result1 = pool.enqueue([]()
//                                 {
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//         std::cout << "Task 1 completed" << std::endl;
//         return 1; });

//     auto result2 = pool.enqueue([]()
//                                 {
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//         std::cout << "Task 2 completed" << std::endl;
//         return 2; });

//     std::cout << "Result 1: " << result1.get() << std::endl;
//     std::cout << "Result 2: " << result2.get() << std::endl;

//     return 0;
// }

// class ThreadPool
// {
// public:
//     ThreadPool(size_t threads);
//     ~ThreadPool();

//     template <class F, class... Args>
//     auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

//     size_t size() const; // Function to get the size of the thread pool

// private:
//     std::vector<std::thread> workers;
//     std::queue<std::function<void()>> tasks;

//     std::mutex queue_mutex;
//     std::condition_variable condition;
//     bool stop;
// };

// // Constructor
// ThreadPool::ThreadPool(size_t threads) : stop(false)
// {
//     for (size_t i = 0; i < threads; ++i)
//         workers.emplace_back(
//             [this]
//             {
//                 for (;;)
//                 {
//                     std::function<void()> task;
//                     {
//                         std::unique_lock<std::mutex> lock(this->queue_mutex);
//                         this->condition.wait(lock, [this]
//                                              { return this->stop || !this->tasks.empty(); });
//                         if (this->stop && this->tasks.empty())
//                             return;
//                         task = std::move(this->tasks.front());
//                         this->tasks.pop();
//                     }
//                     task();
//                 }
//             });
// }

// // Destructor
// ThreadPool::~ThreadPool()
// {
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);
//         stop = true;
//     }
//     condition.notify_all();
//     for (std::thread &worker : workers)
//         worker.join();
// }

// // Add new work item to the pool
// template <class F, class... Args>
// auto ThreadPool::enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
// {
//     using return_type = typename std::result_of<F(Args...)>::type;

//     auto task = std::make_shared<std::packaged_task<return_type()>>(
//         std::bind(std::forward<F>(f), std::forward<Args>(args)...));

//     std::future<return_type> res = task->get_future();
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);
//         if (stop)
//             throw std::runtime_error("enqueue on stopped ThreadPool");
//         tasks.emplace([task]()
//                       { (*task)(); });
//     }
//     condition.notify_one();
//     return res;
// }

// // Function to get the size of the thread pool
// size_t ThreadPool::size() const
// {
//     return workers.size();
// }

// int main()
// {
//     cout << thread::hardware_concurrency() << endl;
//     // ThreadPool pool(4);

//     // auto result1 = pool.enqueue([]()
//     //                             {
//     //     std::this_thread::sleep_for(std::chrono::seconds(1));
//     //     std::cout << "Task 1 completed" << std::endl;
//     //     return 1; });

//     // auto result2 = pool.enqueue([]()
//     //                             {
//     //     std::this_thread::sleep_for(std::chrono::seconds(1));
//     //     std::cout << "Task 2 completed" << std::endl;
//     //     return 2; });

//     // std::cout << "Thread pool size: " << pool.size() << std::endl;

//     // std::cout << "Result 1: " << result1.get() << std::endl;
//     // std::cout << "Result 2: " << result2.get() << std::endl;

//     return 0;
// }

#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
using namespace std;
int main()
{
    cout << thread::hardware_concurrency() << endl;
}
