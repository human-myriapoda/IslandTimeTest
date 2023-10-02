"""
This module contains all the functions needed for querying/storing data from/in a mySQL database

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Load modules 
import mysql.connector

class ManageDatabaseMySQL:
	def __init__(self, user='root', password='*Fuck1984*'):
		self.user = user
		self.password = password

	def queryDatabase(self, database_name: str):
	
		# Change type to str
		if type(database_name) != str: database_name = str(database_name)

		self.database_name = database_name

		# Connect to a database via a mySQL connector
		try:
			self.cnx = mysql.connector.connect(user=self.user, password=self.password, database=self.database_name)

		except:
			self.createDropDatabase(self.database_name, action='create')
			self.cnx = mysql.connector.connect(user=self.user, password=self.password, database=self.database_name)

		# Call the cursor (used to make operations onto the database)
		self.cursor = self.cnx.cursor()

		return self.cursor

	def queryTableFromDatabase(self, table_name: str):

		# Change type to str
		if type(table_name) != str: table_name = str(table_name)

		# Query syntax (all data from a table)
		ostr = "SELECT * FROM {}".format(table_name)

		# Execute query
		self.cursor.execute(ostr)
		table_data = self.cursor.fetchall()

		return table_data

	def queryTableFromDatabaseWhere(self, table_name: str, conditions: list):

		# Change type to str/list
		if type(table_name) != str: table_name = str(table_name)
		if type(conditions) != list: conditions = list(conditions)

		# Original string (to start with)
		query = "SELECT * FROM {} WHERE ".format(table_name)

		# Empty list that will be built to prevent SQL injection 
		values = []

		# Loop on every condition
		for idx, cond in enumerate(conditions):
			
			# Split the condition into [column, operator, value]
			condition_it = cond.split()

			# Append list of values
			values.append(' '.join(condition_it[2:]))

			# Append the variable query with the condition
			if idx == 0: query += '{} {} %s'.format(condition_it[0], condition_it[1])
			else: query += ' AND {} {} %s'.format(condition_it[0], condition_it[1])

		# Execute query
		self.cursor.execute(query, values)
		table_data_where = self.cursor.fetchall()

		return table_data_where

	def createDropTable(self, table_name: str, columns_types=None, action='create'):

		# Action -> CREATE a table into the database
		# Example of columns_types: (id INT AUTO_INCREMENT PRIMARY KEY, island VARCHAR(255), country VARCHAR(255), latitude VARCHAR(255), longitude VARCHAR(255))
		if action == 'create':
			query = '{} TABLE {} {}'.format(action, table_name, columns_types)

		# Action -> DROP a table into the database
		elif action == 'drop':
			query = '{} TABLE {}'.format(action, table_name)

		else:
			raise Exception('Action not recognised')

		self.cursor.execute(query)

	def createDropDatabase(self, database_name: str, action='create'):

		# Connect to mySQL via a connector
		cnx = mysql.connector.connect(user=self.user, password=self.password)

		# Call the cursor (used to make operations)
		cursor = cnx.cursor()

		# Action -> CREATE/DROP a database
		query = '{} DATABASE {}'.format(action, database_name)

		# Execute query
		cursor.execute(query)

	def removeDuplicatesFromTable(self, table_name: str, conditions: list):

		# Change type to str/list
		if type(table_name) != str: table_name = str(table_name)
		if type(conditions) != list: conditions = list(conditions)

		query = 'DELETE FROM {} USING {}, {} e1 WHERE {}.id < e1.id AND '.format(table_name, table_name, table_name, table_name)

		# Loop on every condition
		for idx, cond in enumerate(conditions):
			
			if idx == 0: query += '{}.{} = e1.{}'.format(table_name, cond, cond)
			else: query += ' AND {}.{} = e1.{}'.format(table_name, cond, cond)

		# Execute query
		self.cursor.execute(query)
		self.cnx.commit()

	def storeDictDataTable(self, table_name: str, dict_data: dict):

		# Split the data into placeholders and columns to prevent SQL injection 
		placeholders = ', '.join(['%s'] * len(dict_data))
		columns = ', '.join(dict_data.keys())

		# Add info to query
		query = "INSERT INTO {} ({}) VALUES ({})".format(table_name, columns, placeholders)

		# Execute query
		self.cursor.execute(query, list(dict_data.values()))
		self.cnx.commit()

	def storeSingleDataTable(self, table_name: str, column: str, value):
		pass

	def closeDatabase(self):

		self.cursor.close()
		self.cnx.close()
