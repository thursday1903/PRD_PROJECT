package com.isaac.stock.dao;

import static org.hibernate.criterion.Example.create;

import java.io.Serializable;
import java.util.List;

import org.hibernate.LockMode;
import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.hibernate.exception.ConstraintViolationException;
import org.springframework.orm.hibernate4.SessionFactoryUtils;

import com.isaac.stock.utils.GsonUltilities;
import com.isaac.stock.utils.Logs;

import ch.qos.logback.classic.spi.LoggerComparator;

public class BaseHibernateHome {

	// /private SessionFactory sessionFactory = getSessionFactory();
	final static String alertSystemName = " Database ";

	public static com.isaac.stock.utils.Logs log = new Logs();

	public BaseHibernateHome() {

	}

	protected SessionFactory getSessionFactory() {
		try {
			return null;
		} catch (Exception e) {
			log.error("Could not locate SessionFactory in JNDI", e);
			composeAlertAndSend(e);
			throw new IllegalStateException("Could not locate SessionFactory in JNDI");
		}
	}

	public int persist(Object transientInstance) {
		// log.debug("persisting TblACategories instance");
		int finish = -1;
		Session session = null;
		Transaction trs = null;
		try {
			System.currentTimeMillis();
			session = HibernateUtil.getSessionFactory().openSession();
			System.currentTimeMillis();
			// log.info(">>>>>>>>>>>>>>>>>Total get connection="
			// + (finishProcess - preProcess));
			trs = session.beginTransaction();
			session.persist(transientInstance);

			// log.debug("persist successful");
			trs.commit();
			finish = 0;
		} catch (ConstraintViolationException e) {
			e.printStackTrace();
			finish = -2;
			// TODO: handle exception
		} catch (Exception re) {
			re.printStackTrace();
			String object2Persist = GsonUltilities.toJson(transientInstance);
			log.fatal("persist failed[" + object2Persist + "]", re);
			composeAlertAndSend(re);
		} finally {
			releaseSession(session);
		}
		return finish;
	}

	// public int persistBatch(List<Object> list) {
	// // log.debug("persisting TblACategories instance");
	// int finish = -1;
	// Session session = null;
	// Transaction trs = null;
	// try {
	// System.currentTimeMillis();
	// session = HibernateUtil.getSessionFactory().openSession();
	// System.currentTimeMillis();
	// // log.info(">>>>>>>>>>>>>>>>>Total get connection="
	// // + (finishProcess - preProcess));
	// trs = session.beginTransaction();
	// for (Object object : list) {
	// session.persist(object);
	// }
	// // log.debug("persist successful");
	// trs.commit();
	// finish = 0;
	// } catch (ConstraintViolationException e) {
	// e.printStackTrace();
	// finish = -2;
	// // TODO: handle exception
	// trs.rollback();
	// } catch (Exception re) {
	// re.printStackTrace();
	// // String object2Persist = GsonUltilities.toJson(transientInstance);
	// log.fatal("persist failed", re);
	// trs.rollback();
	// composeAlertAndSend(re);
	// } finally {
	// releaseSession(session);
	// }
	// return finish;
	// }

	public void attachDirty(Object instance) throws Exception {
		// log.debug("attaching dirty TblACategories instance");
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			Transaction trs = session.beginTransaction();
			session.saveOrUpdate(instance);
			trs.commit();
			log.debug("attach successful");
		} catch (Exception re) {
			log.error("attach failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
	}

	public void attachClean(Object instance) throws Exception {
		// log.debug("attaching clean TblACategories instance");
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			session.lock(instance, LockMode.NONE);
			log.debug("attach successful");
		} catch (Exception re) {
			log.error("attach failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
	}

	public int delete(Object persistentInstance) throws Exception {
		log.debug(String.format("deleting %s instance", persistentInstance.getClass().getName()));
		Session session = null;
		int finish = -1;
		Transaction trs = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			trs = session.beginTransaction();
			session.delete(persistentInstance);
			trs.commit();
			finish = 0;
			log.debug("delete successful");
		} catch (Exception re) {
			log.error("delete failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
		return finish;
	}

	public Object merge(Object detachedInstance) throws Exception {
		// log.debug("merging TblACategories instance");
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			// session = sessionFactory.getCurrentSession();
			// session
			Transaction tx = session.beginTransaction();
			Object result = session.merge(detachedInstance);

			log.debug("merge successful");
			tx.commit();
			return result;
		} catch (Exception re) {
			log.error("merge failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
	}

	public int saveOrUpdate(Object detachedInstance) throws Exception {
		// log.debug("merging TblACategories instance");
		Session session = null;
		int finish = -1;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			// session = sessionFactory.getCurrentSession();
			// session
			Transaction tx = session.beginTransaction();
			session.saveOrUpdate(detachedInstance);

			log.debug("merge successful");
			tx.commit();
			finish = 0;
		} catch (ConstraintViolationException e) {
			finish = -2;
			// TODO: handle exception
		} catch (Exception re) {
			log.error("merge failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
		return finish;
	}

	public int update(Object detachedInstance) throws Exception {
		// log.debug("merging TblACategories instance");
		Session session = null;
		int finish = -1;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			// session = sessionFactory.getCurrentSession();
			// session
			Transaction tx = session.beginTransaction();
			session.update(detachedInstance);

			log.debug("merge successful");
			tx.commit();
			finish = 0;
		} catch (ConstraintViolationException e) {
			finish = -2;
			// TODO: handle exception
		} catch (Exception re) {
			log.fatal("merge failed[" + GsonUltilities.toJson(detachedInstance) + "]", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
		return finish;
	}

	public Object findById(Object objId, Object rootObj) throws Exception {
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			Object instance = session.get(rootObj.getClass(), (Serializable) objId);
			if (instance == null) {
				log.debug("get successful, no instance found");
			} else {
				log.debug("get successful, instance found");
			}
			return instance;
		} catch (Exception re) {
			log.error("get failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			try {
				session.close();
			} catch (Exception e) {
				// TODO: handle exception
			}
		}
	}

	public List<Object> findByExample(Object instance) throws Exception {
		// log.debug("finding TblACategories instance by example");
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();

			@SuppressWarnings("unchecked")
			List<Object> results = (List<Object>) session.createCriteria(instance.getClass().getName())
					.add(create(instance)).list();

			return results;
		} catch (Exception re) {
			log.fatal("find by example failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
	}

	public List<Object> listAllObject(Object instance) throws Exception {
		// log.debug("finding TblACategories instance by example");
		Session session = null;
		try {
			System.currentTimeMillis();
			session = HibernateUtil.getSessionFactory().openSession();
			System.currentTimeMillis();

			@SuppressWarnings("unchecked")
			List<Object> results = (List<Object>) session.createCriteria(instance.getClass()).list();

			return results;
		} catch (Exception re) {
			log.fatal("find by example failed", re);
			composeAlertAndSend(re);
			throw re;
		} finally {
			releaseSession(session);
		}
	}

	public Long getSequenceNextvalue(String sequenceName) {
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			String sql = "select " + sequenceName + ".nextval from dual";
			Object uniqueRs = session.createSQLQuery(sql).uniqueResult();
			if (uniqueRs != null)
				return Long.valueOf(uniqueRs.toString());
			return Long.valueOf(1);
		} catch (Exception e) {
			// TODO: handle exception
			log.fatal(e.getLocalizedMessage(), e);
			composeAlertAndSend(e);
			return Long.valueOf(1);
		} finally {
			releaseSession(session);
		}
	}

	public void executeProcedure(String procedureName) {
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			String sql = "CALL " + procedureName + "()";
			session.createSQLQuery(sql).executeUpdate();
		} catch (Exception e) {
			// TODO: handle exception
			log.fatal(e.getLocalizedMessage(), e);
			composeAlertAndSend(e);
		} finally {
			releaseSession(session);
		}
	}

	public void executeProcedure(String procedureName, String param, String value) {
		Session session = null;
		try {
			session = HibernateUtil.getSessionFactory().openSession();
			// String sql = "CALL " + procedureName + "()";
			Query query = session.createSQLQuery("call " + procedureName + " (:" + param + ")").setParameter(param,
					value);
			query.executeUpdate();
			// Object uniqueRs = session.createSQLQuery(sql).executeUpdate();
		} catch (Exception e) {
			// TODO: handle exception
			log.fatal(e.getLocalizedMessage(), e);
			composeAlertAndSend(e);
		} finally {
			releaseSession(session);
		}
	}

	/**
	 * @param e Tao canh? bao va gui cho Admin he thong
	 */
	private void composeAlertAndSend(Throwable e) {
		// AlertBussiness.getInstance().sendEmailAlert(e,
		// MgvConstants.ALERT_CONNECT_DB_FAIL);
		// AlertBussiness.getInstance().sendSMSAlert(MgvConstants.ALERT_CONNECT_DB_FAIL);
	}

	// Cham cuu sau
	// public Long getSequenceNextvalue1(String sequenceName) {
	// Session session = null;
	// try {
	// session = HibernateUtil.getSessionFactory().openSession();
	// String sql = "select " + sequenceName + ".nextval from dual";
	//
	// Object uniqueRs = session.createSQLQuery(sql).uniqueResult();
	// if (uniqueRs != null)
	// return Long.valueOf(uniqueRs.toString());
	// return Long.valueOf(1);
	// } catch (Exception e) {
	// // TODO: handle exception
	// log.fatal(e.getLocalizedMessage(), e);
	// e.printStackTrace();
	// return Long.valueOf(1);
	// } finally {
	// releaseSession(session);
	// }
	// }

	static void releaseSession(Session session) {
		if (session != null)
			try {
				SessionFactoryUtils.closeSession(session);
			} catch (Exception e) {
				// TODO: handle exception
			}
	}
}
