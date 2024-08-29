#ifndef _Vector3_hpp_
#define _Vector3_hpp_

#include <iostream>

template<typename T>
class Vector3 {
public:
    // Constructors
    Vector3(T ix, T iy, T iz);
    Vector3(T ia);
    Vector3();

    // Variables
    T x, y, z;

    // Operators
    Vector3 operator+(const Vector3 &right) const;
    Vector3 operator-(const Vector3 &right) const;
    Vector3 operator*(const Vector3 &right) const;
    Vector3 operator/(const Vector3 &right) const;
    Vector3& operator+=(const Vector3 &right);
    Vector3& operator-=(const Vector3 &right);
    Vector3& operator*=(const Vector3 &right);
    Vector3& operator/=(const Vector3 &right);

    // Methods
    /**
     * @brief Change the vector length to exactly one
     */
    void normalize();
    /**
     * @brief Rotate the vector around the x-axis
     *
     * @param angle Rotation angle in radians.
     */
    void rotateX(T angle);
    /**
     * @brief Rotate the vector around the y-axis
     *
     * @param angle Rotation angle in radians.
     */
    void rotateY(T angle);
    /**
     * @brief Rotate the vector around the z-axis
     *
     * @param angle Rotation angle in radians.
     */
    void rotateZ(T angle);
    /**
     * @brief Return a vector that is the normalized vector of this vector
     * @return This vector normalized
     */
    Vector3 normalized() const;
    /**
     * @brief Calculates the length of the vector and returns it.
     * @return The length of this vector
     */
    T absolute() const;
    /**
     * @brief Calculate the dot product of two vectors.
     * @param left Left hand side of the dot operator.
     * @param right Right hand side of the dot operator.
     * @return Scalar result of the dot operation.
     */
    static T dotProduct(const Vector3 &left, const Vector3 &right);
    /**
     * @brief Calculate the cross product of two vectors.
     * @param left Left hand side of the cross operator.
     * @param right Right hand side of the cross operator.
     * @return Vector result of the cross operation.
     */
    static Vector3 crossProduct(const Vector3 &left, const Vector3 &right);
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector3<T>& vector3);

#endif //_Vector3_hpp_
