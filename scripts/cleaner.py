from bs4 import BeautifulSoup as bs
import re

with open("../data/xml/aviators/Aaron_Buerge.xml") as f:
    soup = bs(f.read(), "lxml")
    init_data = str(soup("text")[0].string)
    init_data = init_data.replace("''", "")
    init_data = init_data.replace("'", "")
    init_data = re.sub(r"{{.+?}}\n", "", init_data)
    init_data = re.sub(r"{{[^}]+}}", "", init_data)
    init_data = re.sub(r"{{[^}]+}}", "", init_data)
    init_data = init_data.replace("[[", "")
    init_data = init_data.replace("]]", "")
    print(init_data)

    # with open("trial.txt", "w+") as out:
    #     out.write(init_data)
    #     out.close()
    f.close()