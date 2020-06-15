package com.isaac.stock.utils;

import org.apache.log4j.Logger;
import org.apache.log4j.xml.DOMConfigurator;

public class Logs {
	Logger logger;
	

	public Logs() {

	}

	public void setClass(Class clazz) {
		// TODO Auto-generated method stub
		logger = Logger.getLogger(clazz);
	}

	public Logs(Class clazz) {
		logger = Logger.getLogger(clazz);
	}

	public Logs(String className) {
		logger = Logger.getLogger(className);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#info(java.lang.String)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#info(java.lang.String)
	 */
	public void info(String input) {
		logger.info(input);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#info(java.lang.String,
	 * java.lang.Throwable)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#info(java.lang.String,
	 * java.lang.Throwable)
	 */
	public void info(String input, Throwable e) {
		logger.info(input, e);
		// AlertBussiness.getInstance().sendEmailAlert(e, 1);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#fatal(java.lang.String)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#fatal(java.lang.String)
	 */
	public void fatal(String input) {
		logger.fatal(input);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#fatal(java.lang.String,
	 * java.lang.Throwable)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#fatal(java.lang.String,
	 * java.lang.Throwable)
	 */
	public void fatal(String input, Throwable e) {
		logger.fatal(input, e);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#debug(java.lang.String)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#debug(java.lang.String)
	 */
	public void debug(String input) {
		logger.debug(input);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#debug(java.lang.String,
	 * java.lang.Throwable)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#debug(java.lang.String,
	 * java.lang.Throwable)
	 */
	public void debug(String input, Throwable e) {
		logger.debug(input, e);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#error(java.lang.String)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#error(java.lang.String)
	 */
	public void error(String input) {
		logger.error(input);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#error(java.lang.String,
	 * java.lang.Throwable)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#error(java.lang.String,
	 * java.lang.Throwable)
	 */
	public void error(String input, Throwable e) {
		logger.error(input, e);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#trace(java.lang.String)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#trace(java.lang.String)
	 */
	public void trace(String input) {
		logger.trace(input);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#trace(java.lang.String,
	 * java.lang.Throwable)
	 */
	/*
	 * (non-Javadoc)
	 * 
	 * @see vn.com.vnptepay.megav.logs.ILog#trace(java.lang.String,
	 * java.lang.Throwable)
	 */
	public void trace(String input, Throwable e) {
		logger.trace(input, e);
	}

	static {
		DOMConfigurator.configure("./config/log4j.xml");
	}

	private static String getMessagePrefix() {
		String stackTrace = Thread.currentThread().getStackTrace()[3].getClassName().toString();
		String function_name = new Exception().getStackTrace()[2].getMethodName();

		return "(" + stackTrace + "." + function_name + ") :";
	}

}
