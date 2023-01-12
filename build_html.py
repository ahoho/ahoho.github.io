from itertools import groupby

import jinja2
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import author, splitname

def parse_authors(record):
    record = author(record) 
    if "author" in record:
        record["author_list"] = []
        for a in record["author"]:
            record["author_list"].append({k: " ".join(v) for k, v in splitname(a).items()})
    return record

def parse_bibtex(infile):
    parser = BibTexParser()
    parser.customization = parse_authors
    parser.common_strings = False
    with open(infile) as infile:
        bib_db = parser.parse_file(infile)

    entries = sorted(bib_db.entries, key=lambda d: -int(d['year']))
    publications = {
        year: list(pubs) for year, pubs in groupby(entries, lambda d: d['year'])
    }
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