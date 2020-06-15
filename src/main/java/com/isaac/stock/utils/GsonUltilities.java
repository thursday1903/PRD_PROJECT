package com.isaac.stock.utils;

import java.util.ArrayList;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class GsonUltilities {
//	static Gson gsonParse = new Gson();
//	static Gson gsonParse = new GsonBuilder().setLenient().disableHtmlEscaping().create();
	static Gson gsonParse = new Gson();
	public static Gson getGsonParse() {
		return gsonParse;
	}

	public static void setGsonParse(Gson gsonParse) {
		GsonUltilities.gsonParse = gsonParse;
	}

	public static String toJson(Object object) {
		return gsonParse.toJson(object);
	}

	@SuppressWarnings("unchecked")
	public static Object fromJson(String jsonString, Class t) {
		return gsonParse.fromJson(jsonString, t);
	}

	public static ArrayList<Object> fromJson(String jsonString,
			java.lang.reflect.Type t) {
		return gsonParse.fromJson(jsonString, t);
	}
}
