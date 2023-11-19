def get_initial_corpus():
    return ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]


def entrypoint(s):
    # 初始条件
    cond1 = False
    cond2 = False

    # 简化的条件判断
    if len(s) > 5 and s[0] == 'a':
        cond1 = True

    # 检查特定模式
    if "buggy" in s:
        cond2 = True

    # 检查是否满足触发 bug 的条件
    if cond1 and cond2:
        print(f"Bug found with string: {s}")
        exit(219)

