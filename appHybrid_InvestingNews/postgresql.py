import psycopg2

HOST='ec2-52-5-1-20.compute-1.amazonaws.com'
DBNAME='d60rbnvko1pi6s'
USER='bkhliiklxjjgce'
PORT='5432'
PASSWORD='e1ec6f00414a7843803c52e448137f9d8213bdb3ffe8cd9bc9517e6b89f2ad01'


def getQuery(query):
    #I also return a result if it exists even if it's INSERT for example in case I want to get the last ID.
    conn = psycopg2.connect(host=HOST,dbname=DBNAME, user=USER, password=PASSWORD,sslmode='require')
    cursor = conn.cursor()
    cursor.execute(query)
    lsResult = cursor.fetchall()
    conn.commit()
    cursor.close()
    conn.close()

    return lsResult

def executeNonQuery(query):
    #"No Returning" means the command doesn't fetch any value back, like Last Id or whatever
    conn = psycopg2.connect(host=HOST,dbname=DBNAME, user=USER, password=PASSWORD,sslmode='require')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
   





