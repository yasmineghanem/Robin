#include <windows.h>
#include <iostream>

int main()
{
    const std::size_t size = 10 * 1000;
    unsigned char *pBuf = nullptr;
    HANDLE hMapFile = NULL, hEvent = NULL, hPythonEvent = NULL;

    // Infinite loop to create/open shared memory and events
    while (true)
    {
        // Create or open the shared memory object
        hMapFile = CreateFileMapping(
            INVALID_HANDLE_VALUE,                    // Use paging file
            NULL,                                    // Default security
            PAGE_READWRITE,                          // Read/Write access
            0,                                       // Maximum object size (high-order DWORD)
            size,                                    // Maximum object size (low-order DWORD)
            TEXT("Local\\MyFixedSizeSharedMemory")); // Name of the mapping object

        if (hMapFile != NULL)
        {
            pBuf = (unsigned char *)MapViewOfFile(
                hMapFile,            // Handle to the map object
                FILE_MAP_ALL_ACCESS, // Read/Write access
                0,
                0,
                size);

            if (pBuf != NULL)
                break;
        }

        // Clean up if creation/opening failed
        if (hMapFile)
            CloseHandle(hMapFile);

        std::cerr << "Waiting to create or open shared memory..." << std::endl;
        Sleep(1000);
    }

    // Create or open the event for signaling Python that data is ready
    while (true)
    {
        hEvent = CreateEvent(
            NULL,                    // Default security attributes
            FALSE,                   // Auto-reset event
            FALSE,                   // Initial state is non-signaled
            TEXT("Local\\MyEvent")); // Name of the event

        if (hEvent != NULL)
            break;

        std::cerr << "Waiting to create or open MyEvent..." << std::endl;
        Sleep(1000);
    }

    // Create or open the event for receiving signal from Python
    while (true)
    {
        hPythonEvent = CreateEvent(
            NULL,                        // Default security attributes
            FALSE,                       // Auto-reset event
            FALSE,                       // Initial state is non-signaled
            TEXT("Local\\PythonEvent")); // Name of the event

        if (hPythonEvent != NULL)
            break;

        std::cerr << "Waiting to create or open PythonEvent..." << std::endl;
        Sleep(1000);
    }

    // Infinite loop to write data to shared memory and wait for Python signal
    while (true)
    {
        // Write data to shared memory
        for (std::size_t i = 0; i < size; ++i)
        {
            pBuf[i] = static_cast<unsigned char>(i % 256);
        }

        // Signal the event to notify Python that data is ready
        if (!SetEvent(hEvent))
        {
            std::cerr << "Failed to set event: " << GetLastError() << std::endl;
            break;
        }

        std::cout << "Data written to shared memory and event signaled. Waiting for Python..." << std::endl;

        // Wait for Python to signal that it has read the data
        if (WaitForSingleObject(hPythonEvent, INFINITE) != WAIT_OBJECT_0)
        {
            std::cerr << "Failed to wait for Python event: " << GetLastError() << std::endl;
            break;
        }

        std::cout << "Python signaled. Continuing..." << std::endl;
    }

    // Clean up
    CloseHandle(hPythonEvent);
    CloseHandle(hEvent);
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);

    return 0;
}
