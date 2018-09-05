import requests
from bs4 import BeautifulSoup as bs

r = requests.get("https://en.wikipedia.org/wiki/List_of_comedians")
soup = bs(r.content, "html.parser")

with open("../json/comdeians.json", "w+", encoding="utf-8") as f:
    f.write("{\n")
    for ul in soup("ul"):
        for li in ul.children:
            try:
                li["class"]
                li["id"]
            except KeyError:
                try:
                    f.write("\t\""+str(li.a.string)+"\": \""+str(li.a["href"])+"\",\n")
                except AttributeError:
                    pass
            except TypeError:
                pass