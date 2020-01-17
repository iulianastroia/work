# -*- coding: utf-8 -*-
import mysql.connector


def db_connect():
    try:
        my_db = mysql.connector.connect(host='37.156.181.224', user='hartapol_admin', password='Parolamea1234',
                                        database='hartapol_account')
        my_cursor = my_db.cursor(buffered=True)
        return my_cursor, my_db
    except mysql.connector.Error as err:
        print("Something went wrong with the connection to the database: {}".format(err))


def insert_values_user(username_value, password_value, email, first_name=None, last_name=None, phone=None):
    insert_query = "insert into user(first_name,last_name,email,phone,username,password) values(%s,%s,%s,%s,%s,%s)"
    val = (first_name, last_name, email, phone, username_value, password_value)
    mycursor, mydb = db_connect()
    mycursor.execute(insert_query, val)
    mydb.commit()
    print(mycursor.rowcount, "record inserted in user table.")
    close_connection(mycursor, mydb)


def insert_null_values(null_field):
    insert_query = "insert into user(" + str(null_field) + ") values(%s)"
    mycursor, mydb = db_connect()
    mycursor.execute(insert_query, null_field)
    mydb.commit()
    print(mycursor.rowcount, "record inserted in user table.")
    close_connection(mycursor, mydb)


def update_values_user(username_value, password_value):
    update_query = "update user set username=%s,password=%s where username=%s"
    val = (username_value, password_value, "user1234")
    mycursor, mydb = db_connect()
    mycursor.execute(update_query, val)
    mydb.commit()
    print(mycursor.rowcount, "record updated in user table.")
    close_connection(mycursor, mydb)


def insert_values_admin(username_value, password_value):
    insert_query = "insert into admin(username,password) values(%s,%s)"
    val = (username_value, password_value)
    mycursor, mydb = db_connect()
    mycursor.execute(insert_query, val)
    mydb.commit()
    print(mycursor.rowcount, "record inserted in admin table.")
    close_connection(mycursor, mydb)


def delete_values(username_value, name):
    sql = "DELETE FROM " + str(name) + " WHERE username = %s"
    adr = (username_value,)
    mycursor, mydb = db_connect()
    mycursor.execute(sql, adr)
    mydb.commit()
    print(mycursor.rowcount, "record(s) deleted from ", name, " table")
    close_connection(mycursor, mydb)


# try:
#     insert_values_user("user", "user", "useruser@yahoo.com")
# except (RuntimeError, TypeError, NameError):
#     print("Please fill out email, username and password(please fill out all mandatory fields)")


# Check if account exists using MySQL
def check_login(username, password):
    mycursor, mydb = db_connect()
    mycursor.execute('SELECT * FROM user WHERE username = %s AND password = %s', (username, password))
    result = mycursor.fetchall()
    if not result:
        print("user and pass DON'T MATCH")
        close_connection(mycursor, mydb)
        return False
    else:
        print("User and pass match in database: ", result)
        close_connection(mycursor, mydb)
        return True


def close_connection(mycursor, mydb):
    mycursor.close()
    mydb.close()


def edit_user(username):
    mycursor, mydb = db_connect()
    mycursor.execute("SELECT * FROM user WHERE username ='" + str(username) + "'")

    result = mycursor.fetchall()
    for x in result:
        first_name = x[1]
        last_name = x[2]
        email = x[3]
        phone = x[4]
        username = x[5]
        password = x[6]
        print("fname", first_name)
        print("lname", last_name)
        print("email", email)
        print("phone", phone)
        print("user", username)
        print("pass", password)
        close_connection(mycursor, mydb)
    return first_name, last_name, email, phone, password
