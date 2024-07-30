import lxml
import requests

URL_TO_COUNTRIES = "https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)"
XPATH_TO_COUNTRIES = "//table[@class='wikitable sortable static-row-numbers plainrowheaders srn-white-background']//tr/td[1]/span/a[1]/@href | //table[@class='wikitable sortable static-row-numbers plainrowheaders srn-white-background']//tr/td[1]/i/a[1]/@href | //table[@class='wikitable sortable static-row-numbers plainrowheaders srn-white-background']//tr/td[1]/span[@class = 'flagicon']/following-sibling::a[1]/@href"

def get_country_names():
    url = requests.get(URL_TO_COUNTRIES)
    doc = lxml.html.fromstring(url.content)
    country_names = []

    for element in doc.xpath(XPATH_TO_COUNTRIES):
        name = str(element).replace(" ", "_")
        if not name.startswith("#"):
            country_names.append(str(element)[6:])

    return country_names