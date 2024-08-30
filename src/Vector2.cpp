#include "../include/Vector2.hpp"
#include <cmath>
#include <immintrin.h>

// Constructor Definitions
template<typename T>
Vector2<T>::Vector2(T ix, T iy) : x(ix), y(iy) {}

template<typename T>
Vector2<T>::Vector2(T ia) : x(ia), y(ia) {}

template<typename T>
Vector2<T>::Vector2() : x(0), y(0) {}

// Add operator
template<typename T>
Vector2<T> Vector2<T>::operator+(const Vector2<T> &right) const {
    Vector2<T> temp(*this);
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, 0, this->y, this->x, this->y, this->x);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, 0, right.y, right.x, right.y, right.x);
        __m256 resultVec = _mm256_add_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, this->y, this->x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.y, right.x);
        __m256d resultVec = _mm256_add_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
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
Vector2<T> Vector2<T>::operator-(const Vector2<T> &right) const {
    Vector2<T> temp(*this);
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, 0, this->y, this->x, this->y, this->x);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, 0, right.y, right.x, right.y, right.x);
        __m256 resultVec = _mm256_sub_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, this->y, this->x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.y, right.x);
        __m256d resultVec = _mm256_sub_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
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
Vector2<T> Vector2<T>::operator*(const Vector2<T> &right) const {
    Vector2<T> temp(*this);
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, 0, this->y, this->x, this->y, this->x);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, 0, right.y, right.x, right.y, right.x);
        __m256 resultVec = _mm256_mul_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, this->y, this->x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.y, right.x);
        __m256d resultVec = _mm256_mul_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
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
Vector2<T> Vector2<T>::operator/(const Vector2<T> &right) const {
    Vector2<T> temp(*this);
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, 0, this->y, this->x, this->y, this->x);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, 0, right.y, right.x, right.y, right.x);
        __m256 resultVec = _mm256_div_ps(leftVec, rightVec);
        temp.x = _mm256_cvtss_f32(resultVec);
        temp.y = _mm256_cvtss_f32(_mm256_shuffle_ps(resultVec, resultVec, _MM_SHUFFLE(1, 1, 1, 1)));
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, this->y, this->x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.y, right.x);
        __m256d resultVec = _mm256_div_pd(leftVec, rightVec);
        temp.x = _mm256_cvtsd_f64(resultVec);
        temp.y = _mm256_cvtsd_f64(_mm256_shuffle_pd(resultVec, resultVec, _MM_SHUFFLE2(1, 1)));
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
Vector2<T>& Vector2<T>::operator+=(const Vector2<T> &right) {
    *this = *this + right;
    return *this;
}

template<typename T>
Vector2<T>& Vector2<T>::operator-=(const Vector2<T> &right) {
    *this = *this - right;
    return *this;
}

template<typename T>
Vector2<T>& Vector2<T>::operator*=(const Vector2<T> &right) {
    *this = *this * right;
    return *this;
}

template<typename T>
Vector2<T>& Vector2<T>::operator/=(const Vector2<T> &right) {
    *this = *this / right;
    return *this;
}

// Normalize the vector
template<typename T>
void Vector2<T>::normalize() {
    if(x != 0 || y != 0){
        T length = absolute();
        *this /= Vector2(length, length);
    }
}

// Rotate the vector by a given angle (in radians)
template<typename T>
void Vector2<T>::rotate(T rotation) {
    T cosRot = std::cos(rotation);
    T sinRot = std::sin(rotation);
    T newX = (x * cosRot) - (y * sinRot);
    y = (x * sinRot) + (y * cosRot);
    x = newX;
}

// Return a new rotated vector
template<typename T>
Vector2<T> Vector2<T>::rotated(T rotation) const {
    T cosRot = std::cos(rotation);
    T sinRot = std::sin(rotation);
    return Vector2(
        (x * cosRot) - (y * sinRot),
        (x * sinRot) + (y * cosRot)
    );
}

// Return a normalized vector
template<typename T>
Vector2<T> Vector2<T>::normalized() const {
    if(x != 0 || y != 0){
        T length = absolute();
        return *this / Vector2(length, length);
    }
    return Vector2<T>(0, 0);
}

// Return the absolute (magnitude) of the vector
template<typename T>
T Vector2<T>::absolute() const {
    return std::sqrt(x * x + y * y);
}

// Return the dot product of two vectors
template<typename T>
T Vector2<T>::dotProduct(const Vector2<T> &left, const Vector2<T> &right) {
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<T, float>) {
        __m256 leftVec = _mm256_set_ps(0, 0, 0, 0, left.y, left.x, left.y, left.x);
        __m256 rightVec = _mm256_set_ps(0, 0, 0, 0, right.y, right.x, right.y, right.x);
        __m256 mulVec = _mm256_mul_ps(leftVec, rightVec);
        float result[8];
        _mm256_storeu_ps(result, mulVec);
        return result[0] + result[1];
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d leftVec = _mm256_set_pd(0, 0, left.y, left.x);
        __m256d rightVec = _mm256_set_pd(0, 0, right.y, right.x);
        __m256d mulVec = _mm256_mul_pd(leftVec, rightVec);
        double result[4];
        _mm256_storeu_pd(result, mulVec);
        return result[0] + result[1];
    } else {
        return left.x * right.x + left.y * right.y;
    }
#else
    return left.x * right.x + left.y * right.y;
#endif
}

// Output stream operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector2<T>& vector2) {
    os << '{' << vector2.x << ',' << vector2.y << '}';
    return os;
}

// Explicit instantiation of templates

// Compile for double
template class Vector2<double>;
template std::ostream& operator<<(std::ostream& os, const Vector2<double>& vector2);

// Compile for float
template class Vector2<float>;
template std::ostream& operator<<(std::ostream& os, const Vector2<float>& vector2);

