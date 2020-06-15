package com.isaac.stock.dao;
// Generated Jun 3, 2020 7:18:17 PM by Hibernate Tools 3.4.0.CR1

import java.util.List;

import com.isaac.stock.entities.TblStockHistory;

/**
 * Home object for domain model class TblStockHistory.
 * 
 * @see tmp.TblStockHistory
 * @author Hibernate Tools
 */

public class TblStockHistoryHome extends BaseHibernateHome {

	public static void main(String[] args) {
		try {
			TblStockHistoryHome tblStockHistoryHome = new TblStockHistoryHome();
			List<Object> list = tblStockHistoryHome.listAllObject(new TblStockHistory());

			System.out.println(list.size());
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
}
