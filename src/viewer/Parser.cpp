#include "Parser.h"

bool Parser::isDouble(QString str)
{
	bool ok;
	str.toDouble(&ok);
	return ok;
}

double Parser::toDouble(QString str)
{
	return str.toDouble();
}

double Parser::toDouble(QString str, bool &ok)
{
	return str.toDouble(&ok);
}

bool Parser::isFloat(QString str)
{
	bool ok;
	str.toFloat(&ok);
	return ok;
}

float Parser::toFloat(QString str)
{
	return str.toFloat();
}

float Parser::toFloat(QString str, bool &ok)
{
	return str.toFloat(&ok);
}

bool Parser::isInt(QString str)
{
	bool ok;
	str.toInt(&ok);
	return ok;
}

int Parser::toInt(QString str)
{
	return str.toInt();
}

int Parser::toInt(QString str, bool &ok)
{
	return str.toInt(&ok);
}