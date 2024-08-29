#include "../include/Vector3.hpp"
#include <cmath>

// TODO: Optimize this using SIMD

template<typename T>
Vector3<T>::Vector3(T ix, T iy, T iz): x(ix), y(iy), z(iz) {}

template<typename T>
Vector3<T>::Vector3(T ia): x(ia), y(ia), z(ia) {}

template<typename T>
Vector3<T>::Vector3(): x(0), y(0), z(0) {}

// Add operator
template<typename T>
Vector3<T> Vector3<T>::operator+(const Vector3<T> &right) const {
    Vector3<T> temp(*this);
    temp += right;
    return temp;
}

// Subtraction operator
template<typename T>
Vector3<T> Vector3<T>::operator-(const Vector3<T> &right) const {
    Vector3<T> temp(*this);
    temp -= right;
    return temp;
}

// Multiplication operator
template<typename T>
Vector3<T> Vector3<T>::operator*(const Vector3<T> &right) const {
    Vector3<T> temp(*this);
    temp *= right;
    return temp;
}

// Division operator
template<typename T>
Vector3<T> Vector3<T>::operator/(const Vector3<T> &right) const {
    Vector3<T> temp(*this);
    temp /= right;
    return temp;
}

template<typename T>
Vector3<T>& Vector3<T>::operator+=(const Vector3<T> &right) {
    x += right.x;
    y += right.y;
    z += right.z;
    return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator-=(const Vector3<T> &right) {
    x -= right.x;
    y -= right.y;
    z -= right.z;
    return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator*=(const Vector3<T> &right) {
    x *= right.x;
    y *= right.y;
    z *= right.z;
    return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator/=(const Vector3<T> &right) {
    if (right.x != 0 && right.y != 0 && right.z != 0) {
        x /= right.x;
        y /= right.y;
        z /= right.z;
    }
    return *this;
}

template<typename T>
void Vector3<T>::normalize() {
    if (x != 0 || y != 0 || z != 0) {
        T length = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
        x /= length;
        y /= length;
        z /= length;
    }
}

template<typename T>
void Vector3<T>::rotateX(T angle) {
    T y_new = y * cos(angle) - z * sin(angle);
    T z_new = y * sin(angle) + z * cos(angle);
    y = y_new;
    z = z_new;
}

template<typename T>
void Vector3<T>::rotateY(T angle) {
    T x_new = x * cos(angle) + z * sin(angle);
    T z_new = -x * sin(angle) + z * cos(angle);
    x = x_new;
    z = z_new;
}

template<typename T>
void Vector3<T>::rotateZ(T angle) {
    T x_new = x * cos(angle) - y * sin(angle);
    T y_new = x * sin(angle) + y * cos(angle);
    x = x_new;
    y = y_new;
}

template<typename T>
Vector3<T> Vector3<T>::normalized() const {
    if (x != 0 || y != 0 || z != 0) {
        T length = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
        return Vector3<T>(
            x / length,
            y / length,
            z / length
        );
    } else {
        return Vector3<T>(0, 0, 0);
    }
}

template<typename T>
T Vector3<T>::absolute() const {
    return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
}

template<typename T>
T Vector3<T>::dotProduct(const Vector3<T> &left, const Vector3<T> &right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

template<typename T>
Vector3<T> Vector3<T>::crossProduct(const Vector3<T> &left, const Vector3<T> &right) {
    return Vector3<T>(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x
    );
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector3<T>& vector3) {
    os << '{' << vector3.x << ',' << vector3.y << ',' << vector3.z << '}';
    return os;
}

// Compile the following classes and functions using the templates

// Compile for double
template class Vector3<double>;
template std::ostream& operator<<(std::ostream& os, const Vector3<double>& vector3);

// Compile for float
template class Vector3<float>;
template std::ostream& operator<<(std::ostream& os, const Vector3<float>& vector3);
