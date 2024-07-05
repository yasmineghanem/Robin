l pool(4);

    // auto result1 = pool.enqueue([]()
    //                             {
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    //     std::cout << "Task 1 completed" << std::endl;
    //     return 1; });

    // auto result2 = pool.enqueue([]()
    //                             {
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    //     std::cout << "Task 2 completed" << std::endl;
    //     return 2; });

    // std::cout << "Thread pool size: " << pool.size() << std::endl;

    // std::cout << "Result 1: " << result1.get() << std::endl;
    // std::cout << "Result 2: " << result2.get() << std::endl;
