import re

def read_file(filepath):
    file = open(filepath, "r")
    content = file.read()
    file.close()
    return content

def save_file(filepath, content):
    file = open(filepath, "w")
    file.write(content)
    file.close()

def fix_version_check(content):
    regex = r"^IS_HIGH_VERSION\s=\s[\s\S]+\s>=\s\[\d+\,\s\d+\,\s\d+\]$"
    return re.sub(regex, "IS_HIGH_VERSION = True", content, 0, re.MULTILINE)

def fix_version_import(content):
    regex = r"from .version import __gitsha__, __version__"
    return re.sub(regex, "", content, 0, re.MULTILINE)

def uncomment_version_check(content):
    regex = r"# IS_HIGH_VERSION = True"
    return re.sub(regex, "IS_HIGH_VERSION = True", content, 0, re.MULTILINE)

basicsr_utils_misc = "basicsr/utils/misc.py"
save_file(basicsr_utils_misc, fix_version_check(read_file(basicsr_utils_misc)))

basicsr_version_import = "basicsr/__init__.py"
save_file(basicsr_version_import, fix_version_import(read_file(basicsr_version_import)))

facelib_yolov5face = "facelib/detection/yolov5face/face_detector.py"
save_file(facelib_yolov5face, fix_version_check(read_file(facelib_yolov5face)))