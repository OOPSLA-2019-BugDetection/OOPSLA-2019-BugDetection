import copy
import os


class generating_longpaths:
    def __init__(self, input_file_name):
        self.input_file_name = input_file_name

    def method_selection(self, string_list):
        level = 0
        methods = []
        method_check = False
        finish_level = 0
        for i in range(len(string_list)):
            if string_list[i].find(":{") != -1:
                level = level + 1
                if not method_check:
                    if string_list[i].find("Method") != -1:
                        method = []
                        method.append(string_list[i].strip())
                        method_check = True
                else:
                    if string_list[i].find("MethodDeclaration;body") != -1:
                        finish_level = level - 1
                    method.append(string_list[i].strip())
            else:
                if string_list[i].find("}") != -1:
                    level = level - 1
                    if method_check:
                        method.append(string_list[i].strip())
                        if level == finish_level:
                            finish_level = 0
                            methods.append(method)
                            method_check = False
                else:
                    if method_check:
                        method.append(string_list[i].strip())
        return methods

    def get_longpaths(self, string_list):
        long_paths = []
        temp = []
        dup = False
        first_half = []
        rest_half = []
        line_first = 0
        line_covers = []
        for i in range(len(string_list)):
            if string_list[i].find(":{") != -1:
                temp.append(string_list[i].strip().split(":{")[0].split(";")[0])
            else:
                if string_list[i].find("}") != -1:
                    temp.pop()
                    dup = False
                else:
                    if not dup:
                        #name = string_list[i].strip().split(":")[0].strip()
                        value = string_list[i].strip().split(";")[2].strip()
                        dup = True
                        if not first_half:
                            first_half = copy.deepcopy(temp)
                            first_half.append(value)
                            line_first = string_list[i].strip().split(";")[3].strip()
                        else:
                            if not rest_half:
                                rest_half = copy.deepcopy(temp)
                                rest_half.append(value)
                                line_rest = string_list[i].strip().split(";")[3].strip()
                                long_path = self.get_longpath(first_half, rest_half)
                                long_paths.append(long_path)
                                if line_first != line_rest:
                                    line_cover = [line_first, line_rest]
                                else:
                                    line_cover = [line_first]
                                line_covers.append(line_cover)
                                first_half = []
                                rest_half = []
        if first_half and not rest_half:
            long_path = self.get_longpath(first_half, first_half)
            long_paths.append(long_path)
        return [long_paths, line_covers]

    def get_longpath (self, first_part, rest_part):
        long_path = first_part[::-1]
        for i in range(len(rest_part)):
            long_path.append(rest_part[i])
        return long_path

    def run(self):
        input_file = open(self.input_file_name, "r")
        lines = input_file.readlines()
        input_file.close()
        methods = self.method_selection(lines)
        long_path_list = []
        for method in methods:
            long_path_set = self.get_longpaths(method)
            long_path_list.append(long_path_set)
        os.remove(self.input_file_name)
        return long_path_list
