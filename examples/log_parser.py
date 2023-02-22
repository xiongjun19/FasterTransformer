# coding=utf8


import sqlite3


def do_parse(f_path):
    try:
        mydb = sqlite3.connect(f_path)
        cursor = mydb.cursor()
        tot_time = _parse_tot_time(cursor)
        kernel_time, nccl_time = _parse_kernel(cursor)
        mem_time = _parse_mem(cursor)
        utilization = (kernel_time - nccl_time) / tot_time
        nccl_ratio = nccl_time / kernel_time
        mem_ratio = mem_time / kernel_time
        cursor.close()
        mydb.close()
        return tot_time, utilization, nccl_ratio, mem_ratio
    except sqlite3.OperationalError as e:
        print(f"Error: there maybe not error in capturing the log of {f_path}")
        print(e)
        return None, None, None, None


def _parse_tot_time(cursor):
    sql = "select duration from ANALYSIS_DETAILS;" # return [(802818559,)]
    items = exec_query(cursor, sql)
    tot = float(items[0][0])
    return tot


def _parse_kernel(cursor):
    sql = "select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0;"
    items = exec_query(cursor, sql)
    ker_time = float(items[0][0])
    sql = "select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0 AND Shortname in (select id from StringIds where  value like '%nccl%');"
    items = exec_query(cursor, sql)
    nccl_time = float(items[0][0])
    return ker_time, nccl_time


def _parse_mem(cursor):
    sql = "select sum(end - start) from CUPTI_ACTIVITY_KIND_MEMSET where deviceId=0;"
    items = exec_query(cursor, sql)
    set_time = float(items[0][0])
    sql = "select sum(end - start) from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0;"
    items = exec_query(cursor, sql)
    cpy_time = float(items[0][0])
    res = set_time + cpy_time
    return res


def exec_query(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()



def test(f_path):
    res = do_parse(f_path)
    print(res)


if __name__ == '__main__':
    import sys
    t_path = sys.argv[1]
    test(t_path)




