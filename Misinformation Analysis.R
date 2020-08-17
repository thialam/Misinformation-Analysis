#Misinformation Analysis | HarvardX Data Science Capstone | Cynthia Lam
true <- tempfile()
download.file("https://github.com/thialam/Misinformation-Analysis/blob/master/True.csv", true)
fake <- tempfile()
download.file("https://github.com/thialam/Misinformation-Analysis/blob/master/Fake.csv", fake)

#Workplace preparation
library(readr)
library(dplyr)
library(tidyverse)
library(caret)
library(stringr)
library(ggplot2)
library(tidytext)
if(!require(tm)) install.packages("tm")
library(tm)
if(!require(textstem)) install.packages("textstem")
library(textstem)
if(!require(wordcloud2)) install.packages("wordcloud2")
library(wordcloud2)
library(randomForest)  
library(naivebayes)

# 1. DATA PREPARATION
url <- "https://github.com/thialam/Misinformation-Analysis/raw/master/True.csv"
True <- read_csv(url)
url <- "https://github.com/thialam/Misinformation-Analysis/raw/master/Fake.csv"
Fake <- read_csv(url)
True <- True %>% mutate(validity = 1) #adding a column for validity where 1 = true
Fake <- Fake %>% mutate(validity = 0) #where 0 = fake
news <- full_join(True,Fake) #combining both dataframes 
news <- news %>% filter(!is.na(text))
set.seed(1, sample.kind = "Rounding")
news <- sample_n(news,20000) #sampling 20000 from the dataset to accommodate the limited computational power
news$validity <- as.factor(news$validity)
news$subject <- as.factor(news$subject)
head(news)

head(news)
# 2. DATA EXPLORATION AND WRANGLING

ggplot(news, aes(x = validity, fill = validity)) + #is the data balanced?
  geom_bar() +
  theme_classic() +
  theme(axis.title = element_text(face = 'bold', size = 15),
        axis.text = element_text(size = 13)) +
  theme(legend.position = 'none')

ggplot(news, aes(x = subject, fill = validity)) + #is the data balanced in every subject?
  geom_bar(position = 'dodge', alpha = 0.6) +
  theme_classic() +
  theme(axis.title = element_text(face = 'bold', size = 15),
        axis.text = element_text(size = 13, angle = 90))

news<-news[-3] #turns out the subjects are completely different in the categories, so we need to delete the subject column as using it as correlating factors produce unreasonably accurate results.
news <- news %>% # Combine 'title' & 'text' column
  unite(col = text ,title, text, sep = ' ')  %>%  
  mutate(ID = as.character(1:nrow(news))) #add a unique ID for each row 

news %>% mutate(length=sapply(strsplit(news$text, " "), length)) %>% #length of the text
  group_by(validity) %>% 
  ggplot(aes(x=length, y=..count.., fill = validity)) +
  geom_density(alpha=0.6) +
  theme_classic() #the distributions overlap, indicating the length does not seem to be a good predictor.


## Text mining 
text <- VCorpus(VectorSource(news$text)) #create a corpus for tm for text mining and cleaning
text <- tm_map(text, content_transformer(tolower))
text <- tm_map(text, removeNumbers)
text <- tm_map(text, content_transformer(str_remove_all), "[[:punct:]]")
text <- tm_map(text, removeWords, stopwords('english'))
text <- tm_map(text, stripWhitespace)

writeLines(as.character(text[[1]])) #inspect to see if all the uninformative words and punctuations have been removed

text <- tm_map(text, content_transformer(lemmatize_strings)) #remove lemmatised strings to return only the meaningful "base" words
writeLines(as.character(text[[1]])) #inspect to see if inflectional endings have been removed and only word bases are retain

dtm <- DocumentTermMatrix(text) #create document term matrix to view the popularity of words and also transition into a clean dataframe
dtm <- removeSparseTerms(dtm, sparse = 0.99) #remove sparse terms
inspect(dtm)
df <- tidy(dtm)

df %>% inner_join(news, by = c('document' = 'ID')) %>% #top terms 
  select(-document) %>%
  group_by(term) %>%
  summarize(freq = sum(count)) %>%
  arrange(desc(freq)) 

realwords <- df %>% inner_join(news, by = c('document' = 'ID')) %>% #top terms for real news
  select(-document) %>%
  filter(validity==1) %>%
  group_by(term) %>%
  summarize(real_freq = sum(count)) %>%
  arrange(desc(real_freq))

fakewords <- df %>% inner_join(news, by = c('document' = 'ID')) %>% #top terms for fake news
    select(-document) %>%
    filter(validity==0) %>%
    group_by(term) %>%
    summarize(fake_freq = sum(count)) %>%
    arrange(desc(fake_freq))

termslist <- fakewords %>% full_join(realwords, by = c('term'='term')) #combine the two lists
termslist <- termslist %>% mutate (ratio = fake_freq/(real_freq+fake_freq)) #find out the OR for fake/real 
sectermslist <- termslist %>%  #there are too many terms and the computer won't be able to manage, so we are only filtering the terms using frequency and ratio thresholds
  arrange(desc(ratio)) %>% 
  filter(!(ratio <=0.65 & ratio >=0.35)) %>%
  filter(!(fake_freq<=1500 & real_freq <=1500)) %>%
  filter(!grepl('reuters', term)) #removing "Reuters" as it makes the data too biased - with an odds ratio of 0.015

df %>% #wordcloud for real news
  inner_join(news, by = c('document' = 'ID')) %>% 
  select(-text) %>%
  filter(validity == 1) %>%
  select(-validity) %>%
  group_by(term) %>%
  summarize(freq = sum(count)) %>%
  arrange(desc(freq)) %>%
  wordcloud2(size = 1.6,  color='random-light')

df %>% #wordcloud for fake news
  inner_join(news, by = c('document' = 'ID')) %>% 
  select(-text) %>%
  filter(validity == 0) %>%
  select(-validity) %>%
  group_by(term) %>%
  summarize(freq = sum(count)) %>%
  arrange(desc(freq)) %>%
  wordcloud2(size = 1.6,  color='random-dark')
  
  
# 3. DATA SET CLEANING AND TRAIN/TEST PREPARATION

news$validity <- as.factor(news$validity)
news1 <- df %>% 
  filter(df$term %in% sectermslist$term) %>% 
  inner_join(news, by = c('document' = 'ID')) %>%
  spread(term,count) %>% #reshape the data so each document takes up one row
  mutate_all(~ifelse(is.na(.), 0, .)) %>%  #turn NA into 0
  select(-c(document, date, text))

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = news1$validity, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- news1[-test_index,]
test_set <- news1[test_index,]
train_set$validity <- as.factor(train_set$validity)
test_set$validity <- as.factor(test_set$validity)
table(train_set$validity) #is it balanced?
table(test_set$validity)

# 4. MODEL TRAINING
## Naive Bayes Model
train_nb <- naive_bayes(validity ~ ., data = train_set)
y_hat0 <- predict(train_nb, newdata = test_set)
confusionMatrix(data = y_hat0, reference = test_set$validity)
mean0 <- mean(y_hat0 == test_set$validity)
results <- data_frame(Method = "Naive Bayes", Accuracy = mean0)
results %>% knitr::kable()

##LDA
set.seed(1, sample.kind = "Rounding")
train_lda <- train(validity ~ ., method = "lda", data = train_set)
y_hat1 <- predict(train_lda, test_set)
mean1 <- mean(y_hat1 == test_set$validity)
confusionMatrix(data = y_hat1, reference = test_set$validity)
results <- bind_rows(results,
                     data_frame(Method="LDA",
                                Accuracy = mean1 ))
results %>% knitr::kable()


##QDA
set.seed(1, sample.kind = "Rounding")
train_qda <- train(validity ~ ., method = "qda", data = train_set)
y_hat2 <- predict(train_qda, test_set)
mean2 <- mean(y_hat2 == test_set$validity)
confusionMatrix(data = y_hat2, reference = test_set$validity)
results <- bind_rows(results,
                     data_frame(Method="QDA",
                                Accuracy = mean2 ))
results %>% knitr::kable()



##GLM
set.seed(1, sample.kind = "Rounding")
train_glm <- train(validity ~ ., method = "glm", data = train_set)
y_hat3 <- predict(train_glm, test_set)
mean3 <- mean(y_hat3 == test_set$validity)
confusionMatrix(data = y_hat3, reference = test_set$validity)
results <- bind_rows(results,
                     data_frame(Method="GLM",
                                Accuracy = mean3 ))
results %>% knitr::kable()
varImp(train_glm)

##KNN
set.seed(1, sample.kind = "Rounding")
k <- 15
train_knn <- knn3(validity ~ ., data=train_set, k = 20)   
y_hat4 <- predict(train_knn,test_set, type = "class") 
mean4 <- mean(y_hat4 == test_set$validity)
mean4
results <- bind_rows(results,
                     data_frame(Method="KNN",
                                Accuracy = mean4 ))
results %>% knitr::kable()


##KNN with cross validation (CV)
set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(validity ~ ., method = "knn", 
                      data = train_set2,
                      tuneGrid = data.frame(k = seq(3, 51, 2)),
                      trControl = control)
ggplot(train_knn_cv, highlight = TRUE)
train_knn_cv$bestTune
y_hat5 <- predict(train_knn_cv,test_set) 
mean5 <- mean(y_hat5 == test_set$validity)

results <- bind_rows(results,
                     data_frame(Method="KNN + Cross Validation",
                                Accuracy = mean5 ))
results %>% knitr::kable()


##Classification Tree
library(rpart)
set.seed(1, sample.kind = "Rounding")
train_rpart <- train(validity ~ ., method ="rpart", 
                     data = train_set, 
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)))

train_rpart$bestTune
y_hat6 <- predict(train_rpart,test_set) 
mean6 <- mean(y_hat6 == test_set$validity)
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)
results <- bind_rows(results,
                     data_frame(Method="Classification Tree",
                                Accuracy = mean6 ))
results %>% knitr::kable()


##RandomForest
set.seed(1, sample.kind = "Rounding")
train_rf<- train(validity ~ ., method ="rf", data = train_set, ntree=50, tuneGrid = data.frame(mtry=seq(1,7,1)))
train_rf$bestTune
y_hat7 <- predict(train_rf,test_set) 
mean7 <- mean(y_hat7 == test_set$validity)
varImp(train_rf) #let's inspect the most important 
results <- bind_rows(results,
                    data_frame(Method="Random Forest",
                                     Accuracy = mean7 ))
results %>% knitr::kable()
impvariables <- varImp(train_rf)$importance
impvariables <- cbind(term = rownames(impvariables), importance = data.frame(impvariables, row.names=NULL))
impvariables <- impvariables %>% arrange(desc(Overall)) %>% top_n(50)


