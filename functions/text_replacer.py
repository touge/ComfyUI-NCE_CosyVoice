

class TextReplacer:
    def __init__(self, input_string, replacement_file=""):
        """
        初始化 TextReplacer 对象并执行替换操作。

        参数:
        input_string (str): 输入的字符串。
        replacement_file (str): 替换规则文件路径。
        """
        self.input_string = input_string
        self.replacement_rules = self.load_replacement_rules_from_txt(replacement_file) if replacement_file else []
        self.result_string = self.replace_phrases()

    def load_replacement_rules_from_txt(self, file_path):
        """
        从 .txt 文件中加载替换规则。

        参数:
        file_path (str): 替换规则文件路径。

        返回:
        list: 替换规则列表，每个规则是一个包含两个元素的元组。
        """
        replacement_rules = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if ',' in line or '，' in line:
                    old, new = line.strip().replace('，', ',').split(',')
                    replacement_rules.append((old, new))
        return replacement_rules

    def replace_phrases(self):
        """
        根据替换规则列表替换字符串中的短语。

        返回:
        str: 替换后的字符串。
        """
        result = self.input_string
        for old, new in self.replacement_rules:
            result = result.replace(old, new)
        return result
