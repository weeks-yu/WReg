#pragma once

//#define SAVE_TEST_INFOS
#define SHOW_Z_INDEX

#include <map>
#include <string>
#include <boost/any.hpp>

class Config {
public:
	static Config* instance();

	template<typename T>
	T get(const std::string param)
	{
		boost::any value = config[param];
		return boost::any_cast<T>(value);
	}

	template<typename T>
	void set(const std::string param, T value)
	{
		config[param] = value;
	}

private:
	std::map<std::string, boost::any> config;
	static Config* _instance;

	Config();
};
