#include "Util.h";
#include "math.h";


void CrossProduct(double a[3], double b[3], double ret[3])
{
    ret[0] = a[1] * b[2] - a[2] * b[1];
    ret[1] = a[2] * b[0] - a[0] * b[2];
    ret[2] = a[0] * b[1] - a[1] * b[0];
}

double DotProduct(double a[3], double b[3])
{
    double result;
    result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    return result;
}

double Normalize(double v[3])
{
    double result;

    result = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    return result;
}

void Rotate(double** rotateMatrix, double u[3], double ret[3]){
    ret[0]=rotateMatrix[0][0]*u[0]+rotateMatrix[0][1]*u[1]+rotateMatrix[0][2]*u[2];
    ret[1]=rotateMatrix[1][0]*u[0]+rotateMatrix[1][1]*u[1]+rotateMatrix[1][2]*u[2];
    ret[2]=rotateMatrix[2][0]*u[0]+rotateMatrix[2][1]*u[1]+rotateMatrix[2][2]*u[2];
}

void RotationMatrix(double angle, double u[3], double rotatinMatrix[3][3])
{
    double norm = Normalize(u);
    
    u[0] = u[0] / norm;
    u[1] = u[1] / norm;
    u[2] = u[2] / norm;

    rotatinMatrix[0][0] = cos(angle) + u[0] * u[0] * (1 - cos(angle));
    rotatinMatrix[0][1] = u[0] * u[1] * (1 - cos(angle) - u[2] * sin(angle));
    rotatinMatrix[0][2] = u[1] * sin(angle) + u[0] * u[2] * (1 - cos(angle));

    rotatinMatrix[1][0] = u[2] * sin(angle) + u[0] * u[1] * (1 - cos(angle));
    rotatinMatrix[1][1] = cos(angle) + u[1] * u[1] * (1 - cos(angle));
    rotatinMatrix[1][2] = -u[0] * sin(angle) + u[1] * u[2] * (1 - cos(angle));
      
    rotatinMatrix[2][0] = -u[1] * sin(angle) + u[0] * u[2] * (1 - cos(angle));
    rotatinMatrix[2][1] = u[0] * sin(angle) + u[1] * u[2] * (1 - cos(angle));
    rotatinMatrix[2][2] = cos(angle) + u[2] * u[2] * (1 - cos(angle));

}

//cal transport
void Calculation3d(double vectorBefore[3], double vectorAfter[3], double rotatinMatrix[3][3])
{
    double  rotationAxis[3];
    double rotationAngle;
    CrossProduct(vectorBefore, vectorAfter, rotationAxis);
    rotationAngle = acos(DotProduct(vectorBefore, vectorAfter) / Normalize(vectorBefore) / Normalize(vectorAfter));
    RotationMatrix(rotationAngle, rotationAxis, rotatinMatrix);
}

void Calculation4d(double vectorBefore[3], double vectorAfter[3], double rotatinMatrix[4][4])
{
    double rotate3d[3][3];
	Calculation3d(vectorBefore,vectorAfter, rotate3d);

	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			rotatinMatrix[i][j] = rotate3d[i][j];

	for(int i = 0; i < 3; i++)
	{
		rotatinMatrix[i][3] = 0;
		rotatinMatrix[3][i] = 0;
	}

	rotatinMatrix[3][3] = 1;
}

void meanEsp(double& Esp, double X, int I)
{
     int Next = I+1;
     double NewEsp =  (X+Esp*I)/(Next);
	 Esp = NewEsp;
}

//增量式方差求法
void meanVar(double& Esp, double &Var, double X, int I)
{
     double C = X - Esp;
     double NewEsp =  (X+Esp*I)/(I+1);
     double NewVar = Var+C*(X-NewEsp);
	 Esp = NewEsp;
	 Var = NewVar;
}

unsigned char MIN(unsigned char a, unsigned char b)
{
	return a < b ? a : b;
}

unsigned char MAX(unsigned char a, unsigned char b)
{
	return a > b ? a : b;
}

void RGB2HSV(unsigned char r, unsigned char g, unsigned char b, double &h, double &s, double &v)
{
    unsigned char min_, max_;
    double delta;

    min_ = MIN(MIN(r, g), b);
    max_ = MAX(MAX(r, g), b);
    v = max_ / 255.0; // v
    delta = max_ - min_;

    if( max_ != 0 )
    {
        s = delta / max_; // s
    }
    else
    {
        // r = g = b = 0 // s = 0, v is undefined
        s = 0;
        h = -1;
        return;
    }

    if( r == max_ )
    {
        h = ( g - b ) / delta; // between yellow & magenta
    }
    else if( g == max_ )
    {
        h = 2 + ( b - r ) / delta; // between cyan & yellow
    }
    else
    {
        h = 4 + ( r - g ) / delta; // between magenta & cyan
    }

    h *= 60; // degrees

    if( h < 0 )
    {
        h += 360;
    }

	h = h * PI / 180.0;
}

void RGB2Lab(unsigned char R, unsigned char G, unsigned char B, double &L, double &a, double &b)
{
	double BLACK = 20;
    double YELLOW = 70;
    double X, Y, Z, fX, fY, fZ;
 
    X = 0.412453*R + 0.357580*G + 0.180423*B;
    Y = 0.212671*R + 0.715160*G + 0.072169*B;
    Z = 0.019334*R + 0.119193*G + 0.950227*B;
    
    X /= (255 * 0.950456);
    Y /=  255;
    Z /= (255 * 1.088754);
    
    if (Y > 0.008856)
    {
     fY = pow(Y, 1.0/3.0);
     L = 116.0*fY - 16.0;
    }
    else
    {
     fY = 7.787*Y + 16.0/116.0;
     L = 903.3*Y;
    }
    
    if (X > 0.008856)
     fX = pow(X, 1.0/3.0);
    else
     fX = 7.787*X + 16.0/116.0;
    
    if (Z > 0.008856)
     fZ = pow(Z, 1.0/3.0);
    else
     fZ = 7.787*Z + 16.0/116.0;
    
    a = 500.0*(fX - fY);
    b = 200.0*(fY - fZ);
    
	/*
    if (L < BLACK) 
    {
     a *= exp((L - BLACK) / (BLACK / 4));
     b *= exp((L - BLACK) / (BLACK / 4));
     L = BLACK;
    }
    if (b > YELLOW)
     b = YELLOW;
	 */

	L = L / 100;
	a = (a+128) / 255;
	b = (b+128) / 255;
}


