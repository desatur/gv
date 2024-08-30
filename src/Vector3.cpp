#include "../include/Vector3.hpp"
#include <cmath>
#include <immintrin.h>

// Constructor Definitions
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
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, z, y, x, 0, 0);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, right.z, right.y, right.x, 0, 0);
        __m256 resultVec = _mm256_add_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
        temp.z = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(2, 2, 2, 2)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, z, y, x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.z, right.y, right.x);
        __m256d resultVec = _mm256_add_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
        temp.z = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(2, 2)));
    } else {
        temp += right;
    }
#else
    temp += right;
#endif
    return temp;
}

// Subtraction operator
template<typename T>
Vector3<T> Vector3<T>::operator-(const Vector3<T> &right) const {
    Vector3<T> temp(*this);
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, z, y, x, 0, 0);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, right.z, right.y, right.x, 0, 0);
        __m256 resultVec = _mm256_sub_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
        temp.z = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(2, 2, 2, 2)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, z, y, x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.z, right.y, right.x);
        __m256d resultVec = _mm256_sub_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
        temp.z = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(2, 2)));
    } else {
        temp -= right;
    }
#else
    temp -= right;
#endif
    return temp;
}

// Multiplication operator
template<typename T>
Vector3<T> Vector3<T>::operator*(const Vector3<T> &right) const {
    Vector3<T> temp(*this);
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, z, y, x, 0, 0);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, right.z, right.y, right.x, 0, 0);
        __m256 resultVec = _mm256_mul_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
        temp.z = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(2, 2, 2, 2)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, z, y, x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.z, right.y, right.x);
        __m256d resultVec = _mm256_mul_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
        temp.z = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(2, 2)));
    } else {
        temp *= right;
    }
#else
    temp *= right;
#endif
    return temp;
}

// Division operator
template<typename T>
Vector3<T> Vector3<T>::operator/(const Vector3<T> &right) const {
    Vector3<T> temp(*this);
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, z, y, x, 0, 0);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, right.z, right.y, right.x, 0, 0);
        __m256 resultVec = _mm256_div_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
        temp.z = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(2, 2, 2, 2)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, z, y, x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.z, right.y, right.x);
        __m256d resultVec = _mm256_div_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
        temp.z = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(2, 2)));
    } else {
        temp /= right;
    }
#else
    temp /= right;
#endif
    return temp;
}

// Compound assignment operators
template<typename T>
Vector3<T>& Vector3<T>::operator+=(const Vector3<T> &right) {
    *this = *this + right;
    return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator-=(const Vector3<T> &right) {
    *this = *this - right;
    return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator*=(const Vector3<T> &right) {
    *this = *this * right;
    return *this;
}

template<typename T>
Vector3<T>& Vector3<T>::operator/=(const Vector3<T> &right) {
    *this = *this / right;
    return *this;
}

// Normalize the vector
template<typename T>
void Vector3<T>::normalize() {
    T length = absolute();
    if (length != 0) {
        *this /= Vector3<T>(length);
    }
}

// Rotate the vector around the X axis
template<typename T>
void Vector3<T>::rotateX(T angle) {
    T cosAngle = std::cos(angle);
    T sinAngle = std::sin(angle);
    T y_new = y * cosAngle - z * sinAngle;
    T z_new = y * sinAngle + z * cosAngle;
    y = y_new;
    z = z_new;
}

// Rotate the vector around the Y axis
template<typename T>
void Vector3<T>::rotateY(T angle) {
    T cosAngle = std::cos(angle);
    T sinAngle = std::sin(angle);
    T x_new = x * cosAngle + z * sinAngle;
    T z_new = -x * sinAngle + z * cosAngle;
    x = x_new;
    z = z_new;
}

// Rotate the vector around the Z axis
template<typename T>
void Vector3<T>::rotateZ(T angle) {
    T cosAngle = std::cos(angle);
    T sinAngle = std::sin(angle);
    T x_new = x * cosAngle - y * sinAngle;
    T y_new = x * sinAngle + y * cosAngle;
    x = x_new;
    y = y_new;
}

// Return the normalized vector
template<typename T>
Vector3<T> Vector3<T>::normalized() const {
    Vector3<T> temp(*this);
    temp.normalize();
    return temp;
}

// Return the magnitude of the vector
template<typename T>
T Vector3<T>::absolute() const {
    return std::sqrt(x * x + y * y + z * z);
}

// Return the dot product of two vectors
template<typename T>
T Vector3<T>::dotProduct(const Vector3<T> &left, const Vector3<T> &right) {
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, left.z, left.y, left.x, 0, 0);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, right.z, right.y, right.x, 0, 0);
        __m256 mulVec = _mm256_mul_ps(leftVec, rightVec);
        float result[8];
        _mm256_storeu_ps(result, mulVec);
        return result[3] + result[4] + result[5];
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, left.z, left.y, left.x);
        __m256d rightVec = _mm256_set_pd(0, right.z, right.y, right.x);
        __m256d mulVec = _mm256_mul_pd(leftVec, rightVec);
        double result[4];
        _mm256_storeu_pd(result, mulVec);
        return result[1] + result[2] + result[3];
    } else {
        return left.x * right.x + left.y * right.y + left.z * right.z;
    }
#else
    return left.x * right.x + left.y * right.y + left.z * right.z;
#endif
}

// Return the cross product of two vectors
template<typename T>
Vector3<T> Vector3<T>::crossProduct(const Vector3<T> &left, const Vector3<T> &right) {
    return Vector3<T>(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x
    );
}

// Output stream operator
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