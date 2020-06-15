package com.isaac.stock.dao;

import org.hibernate.SessionFactory;
import org.springframework.orm.hibernate4.SessionFactoryUtils;

public class HibernateUtil extends SessionFactoryUtils {

	private static final SessionFactory sessionFactory = buildSessionFactory();

	@SuppressWarnings("deprecation")
	private static SessionFactory buildSessionFactory() {
		try {
			return (SessionFactory) ServicesRegister.shareInstance().context
					.getBean("hibernate4AnnotatedSessionFactory");
		} catch (Throwable ex) {
			System.err.println("Initial SessionFactory creation failed." + ex);
			throw new ExceptionInInitializerError(ex);
		}
	}

	public static void releaseSession() {

	}

	public static SessionFactory getSessionFactory() {
		return sessionFactory;
	}

	public static void shutdown() {
		// Close caches and connection pools
		getSessionFactory().close();
	}

}