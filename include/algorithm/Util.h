#ifndef UTIL_H
#define UTIL_H

#define NORM0(x, y, z) max((x),max((y),(z)))
#define NORM1(x, y, z) abs(x) + abs(y) + abs(z)
#define NORM2(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define ISNAN(x) ((x) != (x)) 
#define PI 3.14159265

void CrossProduct(double a[3], double b[3], double ret[3]);

double DotProduct(double a[3], double b[3]);

double Normalize(double v[3]);

void Rotate(double** rotateMatrix, double u[3], double ret[3]);

void RotationMatrix(double angle, double u[3], double rotatinMatrix[3][3]);

//cal transport
void Calculation3d(double vectorBefore[3], double vectorAfter[3], double rotatinMatrix[3][3]);

void Calculation4d(double vectorBefore[3], double vectorAfter[3], double rotatinMatrix[4][4]);

void meanEsp(double& Esp, double X, int I);

void meanVar(double& Esp, double &Var, double X, int I);

void RGB2HSV(unsigned char r, unsigned char g, unsigned char b, double &h, double &s, double &v);

void RGB2Lab(unsigned char R, unsigned char G, unsigned char B, double &L, double &a, double &b);

#endif // UTIL_H
