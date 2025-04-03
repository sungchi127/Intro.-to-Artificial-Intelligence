from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import nltk
import re
def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    #print(preprocessed_text)
    #print(preprocessed_text)
    # Begin your code (Part 0)
    #拔HTML
    preprocessed_text = re.sub("<[^>]+>", "", preprocessed_text).strip()
    #print(preprocessed_text)
    #拔標點符號
    preprocessed_text = re.sub(r'[^\w\s]','',preprocessed_text)
    #print(preprocessed_text)
    #lemmatize
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
    preprocessed_text = word_tokenizer.tokenize(preprocessed_text)
    def lemmatize(word):
        lemma = lemmatizer.lemmatize(word,'v')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word,'n')
        return lemma
    preprocessed_text=[lemmatize(token) for token in preprocessed_text]
    preprocessed_text = ' '.join(preprocessed_text)
    #print(preprocessed_text)
    # End your code
    return preprocessed_text


# Basically  family little boy  Jake  thinks  zombie closet amp parents fighting timeThis movie slower soap opera  suddenly  Jake decides become Rambo kill zombieOK  first  going make film must Decide thriller drama  drama movie watchable Parents divorcing amp arguing like real life Jake closet totally ruins film  expected see BOOGEYMAN similar movie  instead watched drama meaningless thriller spots3 10 well playing parents amp descent dialogs shots Jake  ignore
# Petter Mattei   Love Time Money  visually stunning film watch Mr Mattei offers us vivid portrait human relations movie seems telling us money  power success people different situations encounter This variation Arthur Schnitzler  play theme  director transfers action present time New York different characters meet connect one connected one way  another next person  one seems know previous point contact Stylishly  film sophisticated luxurious look taken see people live world live habitatThe thing one gets souls picture different stages loneliness one inhabits big city exactly best place human relations find sincere fulfillment  one discerns case people encounterThe acting good Mr Mattei  direction Steve Buscemi  Rosario Dawson  Carol Kane  Michael Imperioli  Adrian Grenier  rest talented cast  make characters come aliveWe wish Mr Mattei good luck await anxiously next work

# Basically  family little boy  Jake  thinks  zombie closet amp parents fighting timeThis movie slower soap opera  suddenly  Jake decides become Rambo kill zombieOK  first  going make film must Decide thriller drama  drama movie watchable Parents divorcing amp arguing like real life Jake closet totally ruins film  expected see BOOGEYMAN similar movie  instead watched drama meaningless thriller spots3 10 well playing parents amp descent dialogs shots Jake  ignore
# Petter Mattei   Love Time Money  visually stunning film watch Mr Mattei offers us vivid portrait human relations movie seems telling us money  power success people different situations encounter This variation Arthur Schnitzler  play theme  director transfers action present time New York different characters meet connect one connected one way  another next person  one seems know previous point contact Stylishly  film sophisticated luxurious look taken see people live world live habitatThe thing one gets souls picture different stages loneliness one inhabits big city exactly best place human relations find sincere fulfillment  one discerns case people encounterThe acting good Mr Mattei  direction Steve Buscemi  Rosario Dawson  Carol Kane  Michael Imperioli  Adrian Grenier  rest talented cast  make characters come aliveWe wish Mr Mattei good luck await anxiously next work