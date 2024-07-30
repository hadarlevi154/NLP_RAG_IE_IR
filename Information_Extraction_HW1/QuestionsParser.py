import rdflib
# part 2 - translating #


def natural_language_to_sparql(inputquestion):
    length = len(inputquestion)

    # Not general questions

    if inputquestion.find("How_many") == -1:  ## There's no "how many" in the question

        # Questions 1-2 - "who is the":
        if inputquestion.find("Who_is_the") != -1:

            # Question number 1: Who is the president of <country>?
            if inputquestion.find("president_of") != -1:
                entity1 = inputquestion[24:length - 1]
                relation = "president_of"

            # Question number 2: Who is the prime minister of <country>?
            if inputquestion.find("prime_minister_of") != -1:
                entity1 = inputquestion[29:length - 1]
                relation = "prime_minister_of"

            return "select ?a where {?a <http://example.org/" + relation + "> <http://example.org/" + entity1 + ">. }"

        # Questions 3-6 - "what is the":
        if inputquestion.find("What_is_the") != -1:

            # Question number 3: What is the population of <country>?
            if inputquestion.find("population_of") != -1:
                entity1 = inputquestion[26:length - 1]
                relation = "population"

            # Question number 4: What is the area of <country>?
            if inputquestion.find("area_of") != -1:
                entity1 = inputquestion[20:length - 1]
                relation = "area"

            # Question number 5: What is the form of government in <country>?
            if inputquestion.find("form_of_government_in") != -1:
                entity1 = inputquestion[34:length - 1]
                relation = "government_form"

            # Question number 6: What is the capital of <country>?
            if inputquestion.find("capital_of") != -1:
                entity1 = inputquestion[23:length - 1]
                relation = "capital"

            return "select ?a where {<http://example.org/" + entity1 + "> <http://example.org/" + relation + "> ?a .} ORDER BY ?a"

        # Questions 7+9 - "when was the":
        if inputquestion.find("When_was_the") != -1:

            if inputquestion.find("born") != -1:

                # Question number 7: When was the president of <country> born?
                if inputquestion.find("president_of") != -1:
                    entity1 = inputquestion[26:length - 6]
                    relation = "birth_date"
                    return "select ?date where {?a <http://example.org/president_of> <http://example.org/" + entity1 + ">. " \
                                                "?a <http://example.org/" + relation + "> ?date . }"

                # Question number 9: When was the prime minister of <country> born?
                if inputquestion.find("prime_minister") != -1:
                    entity1 = inputquestion[31:length - 6]
                    relation = "birth_date"
                    return "select ?date where {?a <http://example.org/prime_minister_of> <http://example.org/" + entity1 + ">. " \
                           "?a <http://example.org/" + relation + "> ?date . }"

        # Questions 8+10 - "where was the":
        if inputquestion.find("Where_was_the") != -1:

            if inputquestion.find("born") != -1:

                # Question number 8: Where was the president of <country> born?
                if inputquestion.find("president_of") != -1:
                    entity1 = inputquestion[27:length - 6]
                    relation = "birth_location"
                    return "select ?c where {?a <http://example.org/president_of> <http://example.org/" + entity1 + ">. " \
                                                "?a <http://example.org/" + relation + "> ?c . }"

                # Question number 10: Where was the prime minister of <country> born?
                if inputquestion.find("prime_minister") != -1:
                    entity1 = inputquestion[32:length - 6]
                    relation = "birth_location"
                    return "select ?c where {?a <http://example.org/prime_minister_of> <http://example.org/" + entity1 + ">. " \
                                            "?a <http://example.org/" + relation + "> ?c . }"

        # Question number 11 - "who is <entity>?":
        if (inputquestion.find("Who_is") != -1) and (inputquestion.find("president_of") == -1) and (
                inputquestion.find("prime_minister") == -1):
            entity1 = inputquestion[7:length - 1]
            return "select ?o ?a where {<http://example.org/"+entity1+"> ?o ?a. FILTER(CONTAINS(lcase(str($o)), 'of')). } ORDER BY ?a"

    # General questions

    # Question number 12- How many <government_form1> are also <government_form2>?
    if inputquestion.find("are_also") != -1:
        inputquestion = inputquestion[9:length - 1]
        governmentforms = inputquestion.split("_are_also_")
        return "select (COUNT(?a) AS ?count) where " \
               "{?a <http://example.org/government_form> <http://example.org/" + governmentforms[0] + ">. " \
                "?a <http://example.org/government_form> <http://example.org/" + governmentforms[1] + "> .}"

    # Question number 13- List all countries whose capital name contains the string <str>
    if inputquestion.find("List_all_countries_whose_capital_name_contains_the_string") != -1:
        substring = inputquestion[58:]
        return " select ?country where { ?country <http://example.org/capital> ?capital . FILTER(CONTAINS(lcase(str($capital)), '" + substring + "')). } "

    # Question number 14- How many presidents were born in <country>?
    if inputquestion.find("How_many_presidents_were_born_in") != -1:
        entity1 = inputquestion[33:length - 1]
        return "select (COUNT(*) AS ?count) where" \
               "{ ?x <http://example.org/president_of> ?y ." \
               "?x <http://example.org/birth_location> <http://example.org/" + entity1 + ">.}"

    # Our question- Question number 15- How many prime ministers were born in <country>?
    if inputquestion.find("How_many_prime_ministers_were_born_in") != -1:
        entity1 = inputquestion[38:length - 1]
        return "select (COUNT(*) AS ?count) where" \
               "{ ?x <http://example.org/prime_minister_of> ?y ." \
               "?x <http://example.org/birth_location> <http://example.org/" + entity1 + ">.}"

    # Input question doesnt match any of the questions above
    return "Error"


def question(q):
    #inputquestion = sys.argv[2]
    inputquestion = q
    inputquestion = inputquestion.replace(" ", "_")  # replacing the format from "where is?" to "where_is?"
    sparqlquery = natural_language_to_sparql(inputquestion)

    if sparqlquery == "Error":
        return "whoops... I'm not familiar with the format of the question. Please try again :)"

    graph = rdflib.Graph()
    graph.parse("ontology.nt", format="turtle")
    querylistres = graph.query(sparqlquery)
    querylistres = list(querylistres)

    ### print the answer for question 11
    if (inputquestion.find("Who_is") != -1) and (inputquestion.find("president_of") == -1) and (
                inputquestion.find("prime_minister") == -1):
        res_string = ""
        for j in range(len(querylistres)):
            for i in range(len(querylistres[j])):
                row = querylistres[j][i]  # get row i from query list result.
                entity_with_uri = str(row)
                entity_with_uri = entity_with_uri.split("/")
                entity_without_uri = entity_with_uri[-1]
                # the next 3 code lines are to strip excessive spaces in the names.
                entity_without_uri = entity_without_uri.replace("_", " ")
                entity_without_uri = entity_without_uri.strip()
                entity_without_uri = entity_without_uri.replace(" ", "_")
                res_string += entity_without_uri   # get the entity name without the uri.
                if i % 2 == 0:
                    res_string += " "
            res_string += ", "

        res_string = res_string[0:len(res_string) - 2]
        res_string = res_string.replace("_", " ")
        if len(res_string) > 0:
            return res_string[0].upper() + res_string[1:]


    ### print the answer for questions 12, 14
    if inputquestion.find("How_many") != -1:
        return str(querylistres[0]).split("'")[1]


    ### print the answer for other questions
    if inputquestion.find("How_many") == -1:
        res_string = ""
        for i in range(len(querylistres)):
            row = querylistres[i]  # get row i from query list result.
            entity_with_uri = str(row[0])
            entity_with_uri = entity_with_uri.split("/")
            entity_without_uri = entity_with_uri[-1]
            # the next 3 code lines are to strip excessive spaces in the names.
            entity_without_uri = entity_without_uri.replace("_", " ")
            entity_without_uri = entity_without_uri.strip()
            entity_without_uri = entity_without_uri.replace(" ", "_")
            res_string += entity_without_uri + " "  # get the entity name without the uri.
        names = res_string.split()  # split the string to sort the names lexicographically
        names.sort()
        res_string = ""
        for j in range(len(list(names))):  # build string of names separated by ', '
            res_string += names[j] + ", "
        res_string = res_string[0:len(res_string) - 2]  # remove the last ', ' in the string
        res_string = res_string.replace("_", " ")

        ### print the answer for question 4
        if inputquestion.find("area_of") != -1:
            return res_string + " km squared"

    return res_string

def AnswerQuestion(q):
    print(question(q))

'''
Qs = ["Who is the president of China?", "Who is the president of Portugal?", "Who is the president of Guam?", "Who is the prime minister of Eswatini?", "Who is the prime minister of Tonga?", "What is the population of Isle of Man?", "What is the population of Tokelau?", "What is the population of Djibouti?", "What is the area of Mauritius?", "What is the area of Luxembourg?", "What is the area of Guadeloupe?", "What is the form of government in Argentina?", "What is the form of government in Sweden?", "What is the form of government in Bahrain?", "What is the form of government in North Macedonia?", "What is the capital of Burundi?", "What is the capital of Mongolia?", "What is the capital of Andorra?", "What is the capital of Saint Helena, Ascension and Tristan da Cunha?", "What is the capital of Greenland?", "List all countries whose capital name contains the string hi", "List all countries whose capital name contains the string free", "List all countries whose capital name contains the string alo", "List all countries whose capital name contains the string baba", "How many Absolute monarchy are also Unitary state?", "How many Dictatorship are also Presidential system?", "How many Dictatorship are also Authoritarian?", "How many presidents were born in Iceland? ", "How many presidents were born in Republic of Ireland? ", "When was the president of Fiji born?", "When was the president of United States born?", "Where was the president of Indonesia born?", "Where was the president of Uruguay born?", "Where was the prime minister of Solomon Islands born?", "When was the prime minister of Lesotho born?", "Who is Denis Sassou Nguesso?","Who is David Kabua?"]

Answers = ["Xi Jinping", "Marcelo Rebelo de Sousa", "Joe Biden", "Cleopas Dlamini", "Siaosi Sovaleni", "84,069", "1,499", "921,804", "2,040 km squared", "2,586.4 km squared", "1,628 km squared", "Federal republic, Presidential system, Republic",
"Constitutional monarchy, Parliamentary system, Unitary state", "Islamic state, Parliamentary, Semi-constitutional monarchy, Unitary state", "Parliamentary republic, Unitary state", "Gitega", "Ulaanbaatar", "Andorra la Vella", "Jamestown, Saint Helena", "Nuuk", "Bhutan, India, Moldova, Sint Maarten, United States", "Sierra Leone", "Niue, Tonga", "Eswatini, Ethiopia", "5", "5", "2", "1", "0", "1964-04-20", "1942-11-20", "Indonesia", "Uruguay", "Papua New Guinea", "1961-11-03", "President of Republic of the Congo", "President of Marshall Islands"]


i = 1
for q,ans in zip(Qs, Answers):
    output = question(q)
    if (output != ans):
        print()
        print("Test Failed")
        print("Q: " + q)
        print("Output: " + output)
        print("Answer: " + ans)
        print()
    else:
        print("test " + str(i) + " passed")

    i += 1

'''