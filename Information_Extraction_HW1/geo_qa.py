import sys
from QuestionsParser import *
import lxml.html
from rdflib import Graph, URIRef
from Entities.Country import *
import CountryNames

RDF_URI_PREFIX = "http://example.org/"
URL_TO_COUNTRIES = "https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)"
XPATH_TO_COUNTRIES = "//table[@class='wikitable sortable static-row-numbers plainrowheaders srn-white-background']//tr/td[1]/span/a[1]/@href | //table[@class='wikitable sortable static-row-numbers plainrowheaders srn-white-background']//tr/td[1]/i/a[1]/@href | //table[@class='wikitable sortable static-row-numbers plainrowheaders srn-white-background']//tr/td[1]/span[@class = 'flagicon']/following-sibling::a[1]/@href"

def format_country_name(country_name):
    country_name = country_name.replace("The ", "")
    return urllib.parse.unquote(country_name)

def get_country(country_name):
    return Country(country_name)

def get_countries(doc):
    g = Graph()
    countries = {}
    country_names = CountryNames.get_country_names()

    for name in country_names:
        if not name in countries: #against duplicates
            print(name)
            countries[name] = Country(name)
            add_country_to_ontology(g, countries[name])

    g.serialize(destination="ontology.nt", format="nt", encoding="utf-8")
    return countries


def add_country_to_ontology(g, country):
    country_literal = URIRef(RDF_URI_PREFIX + country.name)
    g.add((country_literal, URIRef(RDF_URI_PREFIX + "population"), URIRef(RDF_URI_PREFIX + country.population)))
    g.add((country_literal, URIRef(RDF_URI_PREFIX + "area"), URIRef(RDF_URI_PREFIX + country.area)))
    if country.capital:
        g.add((country_literal, URIRef(RDF_URI_PREFIX + "capital"), URIRef(RDF_URI_PREFIX + country.capital)))

    for form in country.gov_form:
        g.add((country_literal, URIRef(RDF_URI_PREFIX + "government_form"), URIRef(RDF_URI_PREFIX + form)))

    if country.president:
        president_literal = URIRef(RDF_URI_PREFIX + country.president.name)
        g.add((president_literal, URIRef(RDF_URI_PREFIX + "president_of"), country_literal))
        if country.president.birth_loc:
            g.add((president_literal, URIRef(RDF_URI_PREFIX + "birth_location"),
                   URIRef(RDF_URI_PREFIX + country.president.birth_loc)))
        if country.president.date_of_birth:
            g.add((president_literal, URIRef(RDF_URI_PREFIX + "birth_date"),
                   URIRef(RDF_URI_PREFIX + country.president.date_of_birth)))

    if country.prime_minister:
        prime_minister_literal = URIRef(RDF_URI_PREFIX + country.prime_minister.name)
        g.add((prime_minister_literal, URIRef(RDF_URI_PREFIX + "prime_minister_of"), country_literal))
        if country.prime_minister.birth_loc:
            g.add((prime_minister_literal, URIRef(RDF_URI_PREFIX + "birth_location"),
                   URIRef(RDF_URI_PREFIX + country.prime_minister.birth_loc)))
        if country.prime_minister.date_of_birth:
            g.add((prime_minister_literal, URIRef(RDF_URI_PREFIX + "birth_date"),
                   URIRef(RDF_URI_PREFIX + country.prime_minister.date_of_birth)))

def build_ontology():
    url = requests.get(URL_TO_COUNTRIES)
    doc = lxml.html.fromstring(url.content)
    return get_countries(doc)

if __name__ == '__main__':
    if sys.argv[1] == "create":
        build_ontology()
    if sys.argv[1] == "question":
        AnswerQuestion(sys.argv[2])


