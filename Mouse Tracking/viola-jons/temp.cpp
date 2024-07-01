#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main()
{
    const int iterations = 100000000;
    int intSum = 0;
    double doubleSum = 0.0;

    // Measure time for int operations
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        intSum += i;
    }
    auto end = high_resolution_clock::now();
    auto intDuration = duration_cast<milliseconds>(end - start).count();
    cout << "Time for int operations: " << intDuration << " ms" << endl;

    // Measure time for double operations
    start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        doubleSum += i;
    }
    end = high_resolution_clock::now();
    auto doubleDuration = duration_cast<milliseconds>(end - start).count();
    cout << "Time for double operations: " << doubleDuration << " ms" << endl;

    return 0;
}
