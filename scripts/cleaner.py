from bs4 import BeautifulSoup as bs
from os import walk
import sys

init_path = "../data/xml/" + str(sys.argv[1]) + "/"
dest_path = "../data/text/" + str(sys.argv[1]) + "/"
with open("status_" + str(sys.argv[1]) + ".txt", "w", encoding="utf-8") as status:
    for (dirpath, dirnames, filenames) in walk(init_path):
        for data_file in filenames:
            with open(init_path + data_file, encoding='utf-8') as f:
                soup = bs(f.read(), "lxml")
                init_data = str(soup("text")[0].string)
                init_data = init_data.replace("''", "")
                init_data = init_data.replace("'", "")
                init_data = init_data.replace("|", " ")
                init_data = init_data.replace("[[", "")
                init_data = init_data.replace("]]", "")
                data_list = init_data.split("\n")
                try:
                    end = data_list.index("==References==")
                    start = data_list.index("}}")
                    final_data = "\n".join(data_list[(start+1):end])
                    final_data.strip()
                    with open(dest_path + data_file[:-4] + ".txt", "w+", encoding='utf-8') as out:
                        out.write(final_data)
                        out.close()
                    f.close()
                    status.write(data_file + "\n")
                except ValueError:
                    status.write("empty: " + data_file + "\n")
                    continue