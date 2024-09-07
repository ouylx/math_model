def solve(jobs):
    machine1 = []
    machine2 = []
    time_machine1 = 0
    time_machine2 = 0

    for job in sorted(jobs, reverse=True):
        if time_machine1 <= time_machine2:
            machine1.append(job)
            time_machine1 += job
        else:
            machine2.append(job)
            time_machine2 += job

    completion_time = max(time_machine1, time_machine2)
    return machine1, machine2, completion_time


if __name__ == '__main__':
    # 测试示例
    jobs = [3, 6, 8, 2, 4, 5]
    machine1, machine2, completion_time = solve(jobs)
    print("机器1工件:", machine1)
    print("机器2工件:", machine2)
    print("所有工件完成时间:", completion_time)
