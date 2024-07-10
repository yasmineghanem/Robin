#ifndef MOUSECONTROLLER_H
#define MOUSECONTROLLER_H
#include <vector>
#include <utility>
#include <iostream>
#include <windows.h>

using namespace std;

enum DIRECTION
{
    LEFT = 1,
    RIGHT = 2,
    UP = 3,
    DOWN = 4,
    STRAIGHT = 0
};

class MouseController
{
private:
    // current position off the nose ont te mouse
    pair<int, int> current_position = {-1, -1};
    const double MOVE_THRESHOLD = 10;
    POINT get_current_position();
    void move_relative(int dx, int dy);
    DIRECTION get_horezontal_direction(pair<int, int> p1, pair<int, int> p2);
    DIRECTION get_vertical_direction(pair<int, int> p1, pair<int, int> p2);
    double get_distance(pair<int, int> p1, pair<int, int> p2);

public:
    MouseController();
    double distance_equ(double distance);
    void move_mouse_to(int x, int y);
    void move_mouse_by(DIRECTION, double distance);
    void left_click();
    void right_click();
    void double_click();
    void scroll(int delta);
    void control(vector<pair<int, int>> &landmarks);
    bool isSmile(vector<pair<int, int>> &landmarks);
};
#endif