#include "MouseController.h"
#include <windows.h>
#include "landmark.h"
#include <cmath>
// Constructor
MouseController::MouseController()
{
    // Initialize the current position
    this->current_position = {-1, -1};
}
// distance ^1.7
double MouseController::distance_equ(double distance)
{
    return pow(distance, 1.4);
}
// Function to move the mouse to (x, y) position
// x and y are the screen coordinates
void MouseController::move_mouse_to(int x, int y)
{
    // Create an INPUT structure
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
    // Set the coordinates
    input.mi.dx = (x * 65535) / GetSystemMetrics(SM_CXSCREEN);
    input.mi.dy = (y * 65535) / GetSystemMetrics(SM_CYSCREEN);
    // Send the input
    SendInput(1, &input, sizeof(INPUT));
}
// Get the current position of the mouse
POINT MouseController::get_current_position()
{
    POINT p;
    GetCursorPos(&p);
    return p;
}

// Move the mouse relatively by (dx, dy)
void MouseController::move_relative(int dx, int dy)
{
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    SendInput(1, &input, sizeof(INPUT));
}
void MouseController::move_mouse_by(DIRECTION direction, double distance)
{
    if (direction == STRAIGHT)
    {
        return;
    }
    POINT current_position = get_current_position();
    int current_x = current_position.x;
    int current_y = current_position.y;

    switch (direction)
    {
    case LEFT:
        current_x -= distance_equ(distance);
        break;
    case RIGHT:
        current_x += distance_equ(distance);
        break;
    case UP:
        current_y -= distance_equ(distance);
        break;
    case DOWN:
        current_y += distance_equ(distance);
        break;
    default:
        break;
    }
    int dx = current_x - current_position.x;
    int dy = current_y - current_position.y;
    move_relative(dx, dy);
}
// Function to simulate left mouse click
void MouseController::left_click()
{
    INPUT inputs[2] = {0};
    // Set left button down event
    inputs[0].type = INPUT_MOUSE;
    inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    // Set left button up event
    inputs[1].type = INPUT_MOUSE;
    inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;
    // Send the inputs
    SendInput(2, inputs, sizeof(INPUT));
}

// Function to simulate right mouse click
void MouseController::right_click()
{
    INPUT inputs[2] = {0};

    // Set right button down event
    inputs[0].type = INPUT_MOUSE;
    inputs[0].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;

    // Set right button up event
    inputs[1].type = INPUT_MOUSE;
    inputs[1].mi.dwFlags = MOUSEEVENTF_RIGHTUP;

    // Send the inputs
    SendInput(2, inputs, sizeof(INPUT));
}

// Function to simulate double left mouse click
void MouseController::double_click()
{
    left_click();
    Sleep(100); // Small delay between clicks
    left_click();
}

// Function to simulate mouse scroll
void MouseController::scroll(int delta)
{
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.mouseData = delta;
    input.mi.dwFlags = MOUSEEVENTF_WHEEL;

    // Send the input
    SendInput(1, &input, sizeof(INPUT));
}
bool MouseController::isSmile(vector<pair<int, int>> &landmarks)
{
    if (landmarks.empty())
    {
        return false;
    }

    // Example logic to detect a smile based on landmarks
    // Calculate the distance between the mouth corners
    int dx = abs(landmarks[LEFT_POINT_MOUSE].first - landmarks[RIGHT_POINT_MOUSE].first); // landmarks[LEFT_POINT_MOUSE] and landmarks[RIGHT_POINT_MOUSE] are the mouth corners
    int dy = abs(landmarks[TOP_POINT_MOUSE].second - landmarks[BOTTOM_POINT_MOUSE].second);

    double ratio = static_cast<double>(dx) / dy;
    return ratio >= 4.0;
}
double MouseController::get_distance(pair<int, int> p1, pair<int, int> p2)
{
    return sqrt((p1.first - p2.first) * (p1.first - p2.first) + (p1.second - p2.second) * (p1.second - p2.second));
}
DIRECTION MouseController::get_horezontal_direction(pair<int, int> old_position, pair<int, int> new_position)
{
    if (old_position.first < new_position.first)
    {
        return RIGHT;
    }
    else if (old_position.first > new_position.first)
    {
        return LEFT;
    }
    return STRAIGHT;
}
DIRECTION MouseController::get_vertical_direction(pair<int, int> old_position, pair<int, int> new_position)
{
    if (old_position.second < new_position.second)
    {
        return DOWN;
    }
    else if (old_position.second > new_position.second)
    {
        return UP;
    }
    return STRAIGHT;
}
// Function to control the mouse based on landmarks
void MouseController::control(vector<pair<int, int>> &landmarks)
{
    if (landmarks.empty())
    {
        return;
    }
    if (this->current_position.first == -1 || this->current_position.second == -1)
    {
        this->current_position = landmarks[NOSE_LANDMARK];
        return;
    }
    // if (!this->isSmile(landmarks))
    // {
    //     this->current_position = landmarks[NOSE_LANDMARK];
    //     return;
    // }
    pair<int, int> old_position = this->current_position;
    this->current_position = landmarks[NOSE_LANDMARK];
    double distance = this->get_distance(old_position, this->current_position);
    if (distance < this->MOVE_THRESHOLD)
    {
        return;
    }
    DIRECTION h_direction = this->get_horezontal_direction(old_position, this->current_position);
    DIRECTION v_direction = this->get_vertical_direction(old_position, this->current_position);
    this->move_mouse_by(h_direction, abs(old_position.first - this->current_position.first));
    this->move_mouse_by(v_direction, abs(old_position.second - this->current_position.second));

    // check if left click
    if (landmarks.size() < 5)
    {
        return;
    }
    if (is_left_blink(landmarks))
    {
        this->left_click();
    }
    else if (is_right_blink(landmarks))
    {
        this->right_click();
    }
}
bool MouseController::is_right_blink(vector<pair<int, int>> &landmarks)
{
    // return false;
    // calc the horizontal distane
    auto &p1 = landmarks[5];
    auto &p2 = landmarks[6];
    auto &p3 = landmarks[7];
    auto &p4 = landmarks[8];
    double horizontal_distance = sqrt((p1.first - p2.first) * (p1.first - p2.first) + (p1.second - p2.second) * (p1.second - p2.second));
    double vertical_distance = sqrt((p3.first - p4.first) * (p3.first - p4.first) + (p3.second - p4.second) * (p3.second - p4.second));
    cout << vertical_distance / horizontal_distance << endl;
    return vertical_distance / horizontal_distance <= this->EYE_AR_THRESH;
}
bool MouseController::is_left_blink(vector<pair<int, int>> &landmarks)
{
    // return false;
    // calc the horizontal distane
    auto &p1 = landmarks[1];
    auto &p2 = landmarks[2];
    auto &p3 = landmarks[3];
    auto &p4 = landmarks[4];
    double horizontal_distance = sqrt((p1.first - p2.first) * (p1.first - p2.first) + (p1.second - p2.second) * (p1.second - p2.second));
    double vertical_distance = sqrt((p3.first - p4.first) * (p3.first - p4.first) + (p3.second - p4.second) * (p3.second - p4.second));
    cout << vertical_distance / horizontal_distance << endl;
    return vertical_distance / horizontal_distance <= this->EYE_AR_THRESH;
}
