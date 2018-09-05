import sys
import json
import requests

src = "../json/"+str(sys.argv[1])+".json"
with open(src, "r+", encoding="utf-8") as f:
    srcList = json.loads(f.read())

for link in srcList:
    url = srcList[link].split("/")
    url.insert(2, "Special:Export")
    url.insert(1, "https://en.wikipedia.org")
    url.remove("")
    url = "/".join(url)
    r = requests.get(url)
    fname = "../data/xml/" + str(sys.argv[1]) + "/" + link.replace("'", " ").replace(" ", "_").replace(".", "").replace(",", "").replace("\"", "") + ".xml"
    print(fname)
    with open(fname, "w+", encoding="utf-8") as f:
        f.write(r.text)
        f.close()