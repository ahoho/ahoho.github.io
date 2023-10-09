from itertools import groupby

import jinja2
from bibtexparser import parse_file
import bibtexparser.middlewares as m

def parse_bibtex(infile):
    layers = [
        m.SeparateCoAuthors(True), # Co-authors should be separated as list of strings
        m.SplitNameParts(True) # Individual Names should be split into first, von, last, jr parts
    ]
    bib_db = parse_file(infile, append_middleware=layers)

    entries = sorted(bib_db.entries, key=lambda d: -int(d['year']))
    publications = {}
    # convert to dicts
    for year, pubs in groupby(entries, lambda d: d['year']):
        publications[year] = []
        for p in pubs:
            p = dict(p.items())
            p["author_list"] = [
                {"first": " ".join(auth.first), "last": " ".join(auth.last)}
                for auth in p["author"]
            ]
            publications[year].append(p)
    return publications

if __name__ == "__main__":
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))
    environment.trim_blocks = True
    environment.lstrip_blocks = True
    environment.keep_trailing_newline = False

    pub_template = environment.get_template("pubs.html")
    publications = parse_bibtex("assets/bibliography/papers.bib")
    content = pub_template.render(
        publications=publications,
        last_name="Hoyle",
    )
    with open("index.html", "w") as outfile:
        outfile.write(content)