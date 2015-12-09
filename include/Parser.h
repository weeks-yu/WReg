#ifndef PARSER_H
#define PARSER_H

#include <QString>

class Parser
{
public:
	Parser() {};
	~Parser() {};

	static bool isDouble(QString str);
	static double toDouble(QString str);
	static double toDouble(QString str, bool &ok);

	static bool isInt(QString str);
	static int toInt(QString str);
	static int toInt(QString str, bool &ok);
};

#endif