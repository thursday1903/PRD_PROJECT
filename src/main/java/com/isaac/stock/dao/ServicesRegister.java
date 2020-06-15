package com.isaac.stock.dao;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.FileSystemXmlApplicationContext;

public class ServicesRegister {

	/**
	 * 
	 */
	public ServicesRegister(String configPath) {
		super();
		// TODO Auto-generated constructor stub
		context = new FileSystemXmlApplicationContext("file:" + getFullConfigPath());
		_shareInstance = this;
	}

	public ApplicationContext context;
	private final static String CONFIG_PATH = "/config/Beans.xml";
	private static ServicesRegister _shareInstance = null;
	final static com.isaac.stock.utils.RootConfig root = new com.isaac.stock.utils.RootConfig();

	public static ServicesRegister shareInstance() {
		if (_shareInstance == null) {
			new ServicesRegister(CONFIG_PATH);
		}
		return _shareInstance;
	}

	public Object getInstanceHome(String className) {
		return context.getBean(className);
	}

	protected String getFullConfigPath() {
		// vn.com.vnptepay.config.rootConfig root = new
		// vn.com.vnptepay.config.rootConfig();
		String fullConfigPath = root.getRoot() + CONFIG_PATH;
		// String fullConfigPath = CONFIG_PATH;
		System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>FULL CONFIG PATH " + fullConfigPath);
		return fullConfigPath;
	}

	static {
		shareInstance();
	}

}