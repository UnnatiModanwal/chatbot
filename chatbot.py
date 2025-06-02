# Download necessary NLTK data (run once)
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text_advanced(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation/special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stopwords
    words = text.split()
    #lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
queries = [
    "What is my account balance?",
    "How can I apply for a loan?",
    "Tell me about credit cards",
    "What are your branch timings?",
    "How do I reset my password?"
]

intents = ["balance_inquiry", "loan_application", "credit_card_info", "branch_timings", "password_reset"]

# Apply advanced preprocessing
queries_cleaned = [preprocess_text_advanced(q) for q in queries]
print(queries_cleaned)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(queries_cleaned)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, intents)
responses = {
    "balance_inquiry": "You can check your account balance via our net banking or mobile app.",
    "loan_application": "You can apply for a loan online or visit any of our branches.",
    "credit_card_info": "We offer various credit cards. Please visit our website for details.",
    "branch_timings": "Our branches are open from 9 AM to 5 PM on weekdays.",
    "password_reset": "You can reset your password using the 'Forgot Password' option on login page."
}

def chatbot_response(user_query):
    cleaned = preprocess_text_advanced(user_query)
    vectorized = vectorizer.transform([cleaned])
    predicted_intent = model.predict(vectorized)[0]
    return responses.get(predicted_intent, "Sorry, I didn't understand your query.")

# Test
print(chatbot_response("Can you tell me my balance?"))
print(chatbot_response("I want to know your working hours."))
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Your FAQ data dictionary
faq_data = {
    "how do i apply for a credit card": "You can apply for a credit card online through our website or visit your nearest branch.",
    "how do i reset my internet banking password": "Click on ‘Forgot Password’ on the login page and follow the instructions to reset your password.",
    "how can i open a new account": "You can open a new account online or by visiting any of our branches with valid ID proof.",
    "how do i pay my credit card bill": "You can pay your credit card bill via online banking, mobile app, or at any branch.",
    "what are the bank’s working hours": "Most branches are open from 9:00 AM to 4:00 PM, Monday to Friday, and 9:00 AM to 1:00 PM on Saturdays.",
    "how do i get a copy of my statement": "You can download your statement from online banking or request a printed copy at any branch.",
    "what are the interest rates for savings": "Our current savings account interest rate is 3.5% per annum. Would you like more details?",
    "what should i do if i lose my debit card": "Please call our 24/7 helpline immediately or block your card through the mobile app.",
    "how can i update my address": "You can update your address by uploading proof in your online profile or visiting a branch.",
    "how do i activate my debit card": "You can activate your debit card at any ATM or through our mobile banking app.",
    "how do i check my account balance": "You can check your balance via internet banking, mobile app, or at any ATM.",
    "how do i transfer money to another account": "Use the 'Fund Transfer' option in your online banking or mobile app.",
    "how do i block my credit card": "Call our customer care or use the mobile app to block your credit card instantly.",
    "how do i apply for a loan": "You can apply for a loan online or by visiting your nearest branch.",
    "how do i change my registered mobile number": "Update your mobile number through internet banking or by submitting a request at your branch.",
    "how do i close my account": "Please visit your nearest branch with your ID proof to close your account.",
    "how do i enable international transactions": "Enable international transactions via your mobile app or by contacting customer care.",
    "how do i set up auto-debit for bills": "Set up auto-debit through your online banking under the 'Bill Pay' section.",
    "how do i check my loan status": "Log in to your account online or use the mobile app to check your loan status.",
    "how do i get a new cheque book": "Request a new cheque book via online banking, mobile app, or at your branch.",
    "how do i update my email address": "Update your email address through internet banking or by visiting your branch.",
    "how do i stop a cheque payment": "You can stop cheque payment by logging into your account or contacting customer care.",
    "how do i get my account statement by email": "Register for e-statements through internet banking to receive statements by email.",
    "how do i increase my credit card limit": "Request a limit increase through your mobile app or by calling customer care.",
    "how do i get a duplicate passbook": "Visit your branch and request a duplicate passbook.",
    "how do i dispute a transaction": "Report unauthorized transactions via the mobile app or call customer care immediately.",
    "how do i open a fixed deposit": "Open a fixed deposit online or at any branch.",
    "how do i redeem my credit card reward points": "Redeem your reward points through the credit card section in your online account.",
    "how do i check my credit card statement": "View your credit card statement online or via the mobile app.",
    "how do i change my atm pin": "Change your ATM PIN at any ATM or through the mobile app.",
    "how do i apply for a debit card": "Apply for a debit card online or at your nearest branch.",
    "how do i unlock my internet banking account": "Reset your password online or contact customer care to unlock your account.",
    "how do i update my kyc details": "Update your KYC details by submitting documents at your branch or through the app.",
    "how do i check my emi schedule": "Check your EMI schedule in the loans section of your online account.",
    "how do i download my tax certificate": "Download your tax certificate from the statements section in internet banking.",
    "how do i link my aadhaar to my account": "Link your Aadhaar online, via the app, or by submitting a form at your branch.",
    "how do i get a demand draft": "Request a demand draft online or at any branch.",
    "how do i check my account number": "Find your account number on your passbook, cheque book, or online profile.",
    "how do i get a bank reference letter": "Request a bank reference letter at your branch.",
    "how do i set up standing instructions": "Set up standing instructions through online banking or at your branch.",
    "how do i get my customer id": "Your customer ID is on your welcome letter, passbook, or online profile.",
    "how do i check my account opening status": "Track your account opening status online or by contacting customer care.",
    "how do i close my fixed deposit": "Close your fixed deposit online or at your branch.",
    "how do i get a locker": "Apply for a locker facility at your nearest branch.",
    "how do i check my transaction history": "View your transaction history in your online account or mobile app.",
    "how do i get a mini statement": "Get a mini statement at any ATM or via SMS banking.",
    "how do i update my nominee": "Update your nominee details through online banking or at your branch.",
    "how do i get a loan foreclosure statement": "Request a foreclosure statement through your loan account online or at the branch.",
    "how do i check my fixed deposit maturity date": "Check your FD maturity date in the deposits section of your online account.",
    "how do i apply for net banking": "Register for net banking online or at your branch.",
    "how do i report a phishing email": "Forward phishing emails to report@yourbank.com and do not click any suspicious links.",
    "how do i activate sms alerts": "Activate SMS alerts through your online banking or at your branch.",
    "how do i check forex rates": "View the latest forex rates on our website or mobile app.",
    "how do i get a pre-approved loan offer": "Check your eligibility for pre-approved loans in the offers section of your online account.",
    "how do i open a recurring deposit": "Open a recurring deposit online or at your branch.",
    "how do i update my pan card details": "Update your PAN card details through internet banking or at your branch.",
    "how do i check my account type": "Your account type is mentioned in your passbook, cheque book, or online profile.",
    "how do i get a cancelled cheque": "Issue a cheque and write 'Cancelled' across it; you can get a cheque leaf from your cheque book.",
    "how do i get an interest certificate": "Download your interest certificate from the statements section in internet banking.",
    "how do i check my overdraft limit": "Check your overdraft limit in the accounts section of your online account.",
    "how do i report an atm not dispensing cash": "Report ATM issues via customer care or through your mobile app.",
    "how do i check my account balance by sms": "Send 'BAL' to our SMS banking number from your registered mobile.",
    "how do i get a tds certificate": "Download your TDS certificate from the tax section in internet banking.",
    "how do i check my cheque status": "Check cheque status in your online account or by calling customer care.",
    "how do i get a loan interest certificate": "Download your loan interest certificate from the loan section in internet banking.",
    "how do i set up upi": "Set up UPI through your mobile banking app following the on-screen instructions.",
    "how do i get a bank statement for visa": "Request a visa-specific bank statement at your branch.",
    "how do i check my reward points": "View your reward points in the credit card section of your online account.",
    "how do i get a signature verification letter": "Request a signature verification letter at your branch.",
    "how do i update my marital status": "Update your marital status by submitting proof at your branch.",
    "how do i get a certificate of account balance": "Request a balance certificate at your branch or via online banking.",
    "how do i check my account opening kit status": "Track your kit status online or by contacting customer care.",
    "how do i get a duplicate atm card": "Request a duplicate ATM card through online banking or at your branch.",
    "how do i check my debit card status": "Track your debit card status in the cards section of your online account.",
    "how do i get a bank solvency certificate": "Request a solvency certificate at your branch.",
    "how do i check my account closure status": "Contact customer care or visit your branch to check your account closure status.",
    "how do i apply for a personal loan": "Apply for a personal loan online or at your nearest branch.",
    "how do i check my emi due date": "View your EMI due date in the loans section of your online account.",
    "how do i get a bank guarantee": "Apply for a bank guarantee at your branch.",
    "how do i check my cheque book request status": "Track your cheque book request status in your online account.",
    "how do i get a bank account confirmation letter": "Request a confirmation letter at your branch.",
    "how do i check my credit card application status": "Track your credit card application status online or by contacting customer care.",
    "how do i update my beneficiary details": "Update beneficiary details through online banking or at your branch.",
    "how do i check my fixed deposit interest rate": "Check the current interest rate for fixed deposits on our website or app.",
    "how do i apply for a business loan": "Apply for a business loan online or at your nearest branch.",
    "how do i get a bank statement for tax purposes": "Request a bank statement for tax purposes through online banking or at your branch.",
    "how do i set up mobile banking": "Download our mobile app and register to set up mobile banking.",
    "how do i check my loan eligibility": "Check your loan eligibility online or by contacting customer care.",
    "how do i report a lost cheque book": "Report a lost cheque book immediately by calling customer care or visiting your branch.",
    "how do i get a credit card statement by email": "Register for e-statements to receive your credit card statement by email.",
    "how do i update my contact details": "Update your contact details through internet banking or at your branch.",
    "how do i get a bank statement for visa application": "Request a bank statement specifically for visa application at your branch.",
    "how do i check my account balance at an atm": "Check your account balance at any ATM using your debit card.",
    "how do i get a loan prepayment statement": "Request a prepayment statement for your loan at your branch or online.",
    "how do i set up email alerts": "Set up email alerts for transactions through your online banking.",
    "how do i get a bank statement for loan application": "Request a bank statement for loan application purposes at your branch.",
    "how do i check my credit card reward points expiry": "Check the expiry date of your credit card reward points in your online account.",
    "how do i get a bank statement for scholarship application": "Request a bank statement for scholarship application at your branch.",
    "how do i check my account balance on mobile": "Check your account balance using our mobile app or SMS banking.",
    "how do i get a bank statement for passport application": "Request a bank statement for passport application at your branch.",
    "how do i check my credit card due date": "View your credit card due date in the credit card section of your online account.",
    "how do i get a bank statement for visa interview": "Request a bank statement for visa interview purposes at your branch.",
    "how do i check my account balance on phone": "Call our customer care to check your account balance over the phone.",
    "how do i get a bank statement for government subsidy": "Request a bank statement for government subsidy application at your branch.",
    "how do i check my credit card payment status": "Check your credit card payment status in your online account or mobile app.",
    "how do i get a bank statement for rent agreement": "Request a bank statement for rent agreement purposes at your branch.",
    "how do i check my account balance on internet banking": "Log in to internet banking to check your account balance.",
    "how do i get a bank statement for visa processing": "Request a bank statement for visa processing at your branch.",
    "how do i check my credit card statement by sms": "Send 'CCSTAT' to our SMS banking number to receive your credit card statement.",
    "how do i get a bank statement for education loan": "Request a bank statement for education loan application at your branch.",
    "how do i check my account balance on atm": "Check your account balance at any ATM using your debit card.",
    "how do i get a bank statement for home loan": "Request a bank statement for home loan application at your branch.",
    "how do i check my credit card statement on mobile": "View your credit card statement on our mobile app.",
    "how do i get a bank statement for business loan": "Request a bank statement for business loan application at your branch.",
    "how do i check my account balance on passbook": "Check your account balance in your passbook.",
    "how do i get a bank statement for personal loan": "Request a bank statement for personal loan application at your branch.",
    "how do i check my credit card statement on internet banking": "View your credit card statement by logging into internet banking.",
    "how do i get a bank statement for vehicle loan": "Request a bank statement for vehicle loan application at your branch.",
    "how do i check my account balance on cheque book": "Check your account balance in your cheque book.",
    "how do i get a bank statement for medical loan": "Request a bank statement for medical loan application at your branch.",
    "how do i check my credit card statement on email": "Receive your credit card statement via email by registering for e-statements.",
    "how do i get a bank statement for travel loan": "Request a bank statement for travel loan application at your branch.",
    "how do i check my account balance on mobile app": "Check your account balance using our mobile banking app.",
    "how do i get a bank statement for credit card": "Request a bank statement for credit card application at your branch.",
    "how do i check my credit card statement on phone": "Call customer care to get your credit card statement over the phone.",
    "how do i get a bank statement for debit card": "Request a bank statement for debit card application at your branch.",
    "how do i check my account balance on sms": "Send 'BAL' to our SMS banking number from your registered mobile.",
    "how do i get a bank statement for savings account": "Request a bank statement for savings account application at your branch.",
    "how do i check my credit card statement on sms banking": "Send 'CCSTAT' to our SMS banking number to receive your credit card statement.",
    "how do i get a bank statement for current account": "Request a bank statement for current account application at your branch.",
    "how do i check my account balance on atm machine": "Check your account balance at any ATM machine using your debit card.",
    "how do i get a bank statement for nri account": "Request a bank statement for NRI account application at your branch.",
    "how do i check my credit card statement on mobile app": "View your credit card statement on our mobile app.",
    "how do i get a bank statement for corporate account": "Request a bank statement for corporate account application at your branch.",
    "how do i check my account balance on passbook update": "Check your account balance when your passbook is updated at the branch.",
    "how do i get a bank statement for trust account": "Request a bank statement for trust account application at your branch.",
    "how do i check my credit card statement on internet banking portal": "View your credit card statement by logging into the internet banking portal.",
    "how do i get a bank statement for savings account statement": "Request a bank statement for savings account statement at your branch.",
    "how do i check my account balance on cheque book leaf": "Check your account balance on your cheque book leaf.",
    "how do i get a bank statement for fixed deposit": "Request a bank statement for fixed deposit application at your branch.",
    "how do i check my credit card statement on email statement": "Receive your credit card statement via email by registering for e-statements.",
    "how do i get a bank statement for recurring deposit": "Request a bank statement for recurring deposit application at your branch.",
    "how do i check my account balance on mobile banking": "Check your account balance using our mobile banking service.",
    "how do i get a bank statement for savings account passbook": "Request a bank statement for your savings account passbook at your branch.",
    "how do i check my credit card pin": "You can view or reset your credit card PIN using the mobile app or internet banking.",
    "how do i get a bank statement for overdraft account": "Request a bank statement for overdraft account application at your branch.",
    "how do i check my loan account number": "Find your loan account number in your loan documents or online account.",
    "how do i get a bank statement for partnership account": "Request a bank statement for partnership account application at your branch.",
    "how do i check my last five transactions": "View your last five transactions in your mobile app or at any ATM.",
    "how do i get a bank statement for joint account": "Request a bank statement for joint account application at your branch.",
    "how do i check my fixed deposit account number": "Find your FD account number in your FD receipt or online account.",
    "how do i get a bank statement for minor account": "Request a bank statement for minor account application at your branch.",
    "how do i check my recurring deposit account number": "Find your RD account number in your RD receipt or online account.",
    "how do i get a bank statement for overdraft facility": "Request a bank statement for overdraft facility at your branch.",
    "how do i check my loan repayment schedule": "View your loan repayment schedule in your online account.",
    "how do i get a bank statement for club account": "Request a bank statement for club account application at your branch.",
    "how do i check my emi amount": "Check your EMI amount in the loans section of your online account.",
    "how do i get a bank statement for society account": "Request a bank statement for society account application at your branch.",
    "how do i check my account opening date": "Your account opening date is available in your account profile online or on your welcome letter.",
    "how do i get a bank statement for huf account": "Request a bank statement for HUF account application at your branch.",
    "how do i check my account balance in us dollars": "Check your foreign currency account balance in your online account.",
    "how do i get a bank statement for salary account": "Request a bank statement for salary account application at your branch.",
    "how do i check my account balance in gbp": "Check your GBP account balance in your online account.",
    "how do i get a bank statement for senior citizen account": "Request a bank statement for senior citizen account at your branch.",
    "how do i check my account balance in euro": "Check your Euro account balance in your online account.",
    "how do i get a bank statement for student account": "Request a bank statement for student account application at your branch.",
    "how do i check my account balance in yen": "Check your Yen account balance in your online account.",
    "how do i get a bank statement for pension account": "Request a bank statement for pension account application at your branch.",
    "how do i check my account balance in aud": "Check your AUD account balance in your online account.",
    "how do i get a bank statement for savings plus account": "Request a bank statement for savings plus account at your branch.",
    "how do i check my account balance in cad": "Check your CAD account balance in your online account.",
    "how do i get a bank statement for woman account": "Request a bank statement for woman account application at your branch.",
    "how do i check my account balance in sgd": "Check your SGD account balance in your online account.",
    "how do i get a bank statement for premium account": "Request a bank statement for premium account at your branch.",
    "how do i check my account balance in chf": "Check your CHF account balance in your online account.",
    "how do i get a bank statement for platinum account": "Request a bank statement for platinum account at your branch.",
    "how do i check my account balance in zar": "Check your ZAR account balance in your online account.",
    "how do i get a bank statement for diamond account": "Request a bank statement for diamond account at your branch.",
    "how do i check my account balance in sek": "Check your SEK account balance in your online account.",
    "how do i get a bank statement for gold account": "Request a bank statement for gold account at your branch.",
    "how do i check my account balance in nok": "Check your NOK account balance in your online account.",
    "how do i get a bank statement for silver account": "Request a bank statement for silver account at your branch.",
    "how do i check my account balance in hkd": "Check your HKD account balance in your online account.",
    "how do i get a bank statement for kids account": "Request a bank statement for kids account at your branch.",
    "how do i check my account balance in nzd": "Check your NZD account balance in your online account.",
    "how do i get a bank statement for youth account": "Request a bank statement for youth account at your branch.",
    "How do i activate mobile banking":"Register and activate mobile banking through your banks app or at your branch.",
    "How do i reset my online banking password ":"Reset your password using the “Forgot Password” option on your bank’s login page.",
    "How do i apply for a personal loan" : "Apply for a personal loan online or at your branch.",
    "How do i check my loan balance" :"Check your loan balance in your online banking account or contact your branch.",
    "How do i renew my fixed deposit":"Renew your fixed deposit online or at your branch.",
    "How do i know about froud": "Check your bank statement for tracking transection statement"
    }

# 1. Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Preprocess all questions
questions = list(faq_data.keys())
preprocessed_questions = [preprocess_text(q) for q in questions]

# 2. Vectorize questions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

# 3. Search function
def search_faq(query, top_n=3):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get indices of top N similar questions
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        question = questions[idx]
        answer = faq_data[question]
        score = cosine_similarities[idx]
        results.append((question, answer, score))
    return results

# Example search
query = input('enetr your query here: ')
results = search_faq(query)

for i, (q, a, score) in enumerate(results, 1):
    print(f"{i}. Q: {q} (Score: {score:.3f})\n   A: {a}\n")
