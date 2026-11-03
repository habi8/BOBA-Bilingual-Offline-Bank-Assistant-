"""Create a reviewable multilingual fine-tuning set from bank-faq.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAQ_PATH = ROOT / "bank-faq.csv"
QUESTIONS_PATH = ROOT / "test-questions.csv"
OUTPUT_PATH = ROOT / "finetune_examples.jsonl"

# Each record is (FAQ ID, language, paraphrased question, faithful answer).
# The user reviewed and approved these paraphrases and translations.
EXAMPLES = [
    ("1", "English", "What should I bring to open a savings account?", "You need your NID or passport, two passport-size photos, and nominee details. A TIN certificate is optional. The minimum opening deposit is BDT 500."),
    ("1", "Bangla", "সেভিংস অ্যাকাউন্ট খুলতে কী কী কাগজপত্র লাগবে?", "সেভিংস অ্যাকাউন্ট খুলতে আপনার NID বা পাসপোর্ট, দুই কপি পাসপোর্ট সাইজের ছবি এবং নমিনির তথ্য লাগবে। TIN সার্টিফিকেট ঐচ্ছিক। অ্যাকাউন্ট খোলার ন্যূনতম জমা BDT 500।"),
    ("1", "Banglish", "Savings account khulte ki ki document lagbe?", "Savings account khulte apnar NID ba passport, dui copy passport-size chobi, ebong nominee-r details lagbe. TIN certificate optional. Account kholar minimum deposit BDT 500."),
    ("2", "English", "How much should I keep in my savings account each month?", "Keep an average monthly balance of BDT 1,000. If it falls below this, a quarterly fee of BDT 150 plus VAT is charged."),
    ("2", "Bangla", "সেভিংস অ্যাকাউন্টে মাসে গড়ে কত টাকা রাখতে হবে?", "সেভিংস অ্যাকাউন্টে মাসিক গড় ব্যালেন্স BDT 1,000 রাখুন। এর নিচে হলে প্রতি তিন মাসে BDT 150 ফি এবং VAT প্রযোজ্য।"),
    ("2", "Banglish", "Savings account e monthly average balance koto rakhte hobe?", "Savings account-e monthly average balance BDT 1,000 rakhte hobe. Er niche hole proti quarter-e BDT 150 fee plus VAT charge hoy."),
    ("3", "English", "How much is the debit card's yearly charge?", "The first year is free. From the second year the annual fee is BDT 500."),
    ("3", "Bangla", "ডেবিট কার্ডের বার্ষিক চার্জ কত?", "প্রথম বছর কোনো ফি নেই। দ্বিতীয় বছর থেকে বার্ষিক ফি BDT 500।"),
    ("3", "Banglish", "Debit card-er yearly fee koto?", "Prothom bochor free. Ditiyo bochor theke annual fee BDT 500."),
    ("4", "English", "What should I do right away if my bank card is stolen?", "Call the 24-hour hotline 16999 immediately to block the card. A replacement card can be collected from your branch within 7 working days for a fee of BDT 300."),
    ("4", "Bangla", "ব্যাংক কার্ড চুরি হলে সঙ্গে সঙ্গে কী করব?", "কার্ডটি ব্লক করতে সঙ্গে সঙ্গে ২৪ ঘণ্টার হটলাইন 16999-এ কল করুন। BDT 300 ফি দিয়ে ৭ কর্মদিবসের মধ্যে শাখা থেকে প্রতিস্থাপন কার্ড নিতে পারবেন।"),
    ("4", "Banglish", "Bank card churi hole shathe shathe ki korbo?", "Card block korte shathe shathe 24-hour hotline 16999-e call korun. BDT 300 fee diye 7 working day-er moddhe branch theke replacement card nite parben."),
    ("5", "English", "Where can I change my card PIN?", "Open the NB Go app, go to Settings, then Card, then Reset PIN. You can also reset it at any Nirbhor Bank ATM using an OTP sent to your registered number."),
    ("5", "Bangla", "কার্ডের PIN কীভাবে নতুন করে সেট করব?", "NB Go অ্যাপে Settings, তারপর Card, তারপর Reset PIN-এ যান। নিবন্ধিত নম্বরে পাঠানো OTP ব্যবহার করে Nirbhor Bank-এর যেকোনো ATM থেকেও PIN রিসেট করতে পারবেন।"),
    ("5", "Banglish", "Card-er PIN notun kore set korbo kibhabe?", "NB Go app-e Settings, tarpor Card, tarpor Reset PIN-e jan. Registered number-e pathano OTP diye Nirbhor Bank-er jekono ATM thekeo PIN reset korte parben."),
    ("6", "English", "How do I sign up for NB Go?", "Download NB Go, choose Register, enter your account number and registered mobile number, then enter the OTP we send you and set a 6-digit passcode."),
    ("6", "Bangla", "NB Go অ্যাপে নতুন করে নিবন্ধন করব কীভাবে?", "NB Go ডাউনলোড করে Register নির্বাচন করুন। অ্যাকাউন্ট নম্বর ও নিবন্ধিত মোবাইল নম্বর দিন, পাঠানো OTP লিখুন, তারপর ৬ সংখ্যার পাসকোড সেট করুন।"),
    ("6", "Banglish", "NB Go app-e registration korar upay ki?", "NB Go download kore Register select korun. Account number ebong registered mobile number din, pathano OTP likhun, tarpor 6-digit passcode set korun."),
    ("7", "English", "What are the per-day limits for sending money?", "Transfers to Nirbhor Bank accounts: up to BDT 10 lakh per day. Transfers to other banks through BEFTN: up to BDT 5 lakh per day. RTGS transfers start from BDT 1 lakh."),
    ("7", "Bangla", "প্রতিদিন টাকা পাঠানোর সর্বোচ্চ সীমা কত?", "Nirbhor Bank অ্যাকাউন্টে দিনে সর্বোচ্চ BDT 10 লাখ পাঠানো যায়। BEFTN-এর মাধ্যমে অন্য ব্যাংকে দিনে সর্বোচ্চ BDT 5 লাখ পাঠানো যায়। RTGS ট্রান্সফার BDT 1 লাখ থেকে শুরু।"),
    ("7", "Banglish", "Protidin taka pathanor maximum limit koto?", "Nirbhor Bank account-e dine maximum BDT 10 lakh pathano jay. BEFTN diye onno bank-e dine maximum BDT 5 lakh pathano jay. RTGS transfer BDT 1 lakh theke shuru."),
    ("8", "English", "How long does a BEFTN payment take to arrive?", "A BEFTN transfer is credited by the next working day, as long as it is sent before 3:00 PM."),
    ("8", "Bangla", "BEFTN-এ পাঠানো টাকা কখন অন্য ব্যাংকে জমা হয়?", "বিকেল ৩টার আগে পাঠালে BEFTN ট্রান্সফার পরবর্তী কর্মদিবসের মধ্যে জমা হয়।"),
    ("8", "Banglish", "BEFTN-e pathano taka onno bank-e kokhon pouchhay?", "Bikel 3tar age pathale BEFTN transfer porer working day-er moddhe credit hoy."),
    ("9", "English", "What is the maximum cash I can take from an ATM?", "You can withdraw up to BDT 50,000 per day per card, and up to BDT 20,000 in a single transaction."),
    ("9", "Bangla", "এক দিনে ATM থেকে সর্বোচ্চ কত টাকা তুলতে পারি?", "প্রতি কার্ডে দিনে সর্বোচ্চ BDT 50,000 তোলা যায় এবং একবারের লেনদেনে সর্বোচ্চ BDT 20,000 তোলা যায়।"),
    ("9", "Banglish", "Ek dine ATM theke maximum koto taka tulte parbo?", "Proti card-e dine maximum BDT 50,000 tola jay, ar ekta transaction-e maximum BDT 20,000 tola jay."),
    ("10", "English", "Which days and times are bank branches open?", "Branches are open Sunday to Thursday, 10:00 AM to 4:00 PM. They are closed on Friday and Saturday and on government holidays."),
    ("10", "Bangla", "ব্যাংকের শাখা কোন দিন ও কখন খোলা থাকে?", "শাখা রবিবার থেকে বৃহস্পতিবার সকাল ১০টা থেকে বিকেল ৪টা পর্যন্ত খোলা থাকে। শুক্রবার, শনিবার এবং সরকারি ছুটির দিনে বন্ধ থাকে।"),
    ("10", "Banglish", "Bank branch kon kon din, koyta theke koyta porjonto khola?", "Branch Robibar theke Brihoshpotibar shokal 10ta theke bikel 4ta porjonto khola. Shukrobar, Shonibar ebong government holiday-te bondho thake."),
    ("11", "English", "How can I download a recent account statement?", "The last 6 months of statements are free in the NB Go app. A printed, stamped statement from a branch costs BDT 100 per statement."),
    ("11", "Bangla", "সাম্প্রতিক অ্যাকাউন্ট স্টেটমেন্ট কোথায় পাব?", "NB Go অ্যাপে গত ৬ মাসের স্টেটমেন্ট বিনামূল্যে পাবেন। শাখা থেকে প্রিন্ট করা ও সিলযুক্ত প্রতিটি স্টেটমেন্টের দাম BDT 100।"),
    ("11", "Banglish", "Recent account statement kibhabe pabo?", "NB Go app-e shesh 6 mash-er statement free. Branch theke printed, stamped protiti statement-er fee BDT 100."),
    ("12", "English", "How can I update the mobile number linked to my account?", "Visit your branch with your NID and fill in the update form, or update it in the NB Go app after face verification."),
    ("12", "Bangla", "অ্যাকাউন্টে দেওয়া মোবাইল নম্বর বদলাব কীভাবে?", "NID নিয়ে শাখায় গিয়ে আপডেট ফর্ম পূরণ করুন, অথবা face verification-এর পর NB Go অ্যাপে নম্বরটি আপডেট করুন।"),
    ("12", "Banglish", "Account-er shathe jukto mobile number change korbo kibhabe?", "NID niye branch-e giye update form puron korun, ba face verification-er por NB Go app-e number update korun."),
    ("13", "English", "How do I reactivate an account that has gone dormant?", "An account with no customer transaction for 12 months becomes dormant. Visit your branch with your NID and make a deposit to reactivate it."),
    ("13", "Bangla", "অ্যাকাউন্ট ডরম্যান্ট হলে আবার চালু করার নিয়ম কী?", "১২ মাস গ্রাহকের কোনো লেনদেন না হলে অ্যাকাউন্ট ডরম্যান্ট হয়। NID নিয়ে শাখায় যান এবং অ্যাকাউন্টটি পুনরায় চালু করতে টাকা জমা দিন।"),
    ("13", "Banglish", "Account dormant hole abar active korbo kibhabe?", "12 mash customer transaction na hole account dormant hoy. NID niye branch-e jan ebong account abar active korte deposit korun."),
    ("14", "English", "Can two people open a shared bank account?", "Yes. A joint account can have up to 2 holders, and both holders must sign the account opening documents."),
    ("14", "Bangla", "দুজন মিলে যৌথ ব্যাংক অ্যাকাউন্ট খোলা যাবে?", "হ্যাঁ। যৌথ অ্যাকাউন্টে সর্বোচ্চ ২ জন হিসাবধারী থাকতে পারেন এবং দুজনকেই অ্যাকাউন্ট খোলার কাগজে স্বাক্ষর করতে হবে।"),
    ("14", "Banglish", "Dujon mile joint bank account khola jabe?", "Hyan. Joint account-e maximum 2 jon holder thakte paren, ebong account opening document-e dujon-kei sign korte hobe."),
    ("15", "English", "What do I need to change the nominee on my account?", "Submit a nominee change form at your branch with the new nominee's NID and photo. Both the account holder and the nominee must sign."),
    ("15", "Bangla", "অ্যাকাউন্টের নমিনি পরিবর্তন করতে কী জমা দিতে হবে?", "নতুন নমিনির NID ও ছবি দিয়ে শাখায় নমিনি পরিবর্তনের ফর্ম জমা দিন। অ্যাকাউন্টধারী ও নমিনি—দুজনকেই স্বাক্ষর করতে হবে।"),
    ("15", "Banglish", "Account-er nominee change korte ki lagbe?", "Notun nominee-r NID ebong chobi diye branch-e nominee change form joma din. Account holder ebong nominee dujon-kei sign korte hobe."),
    ("16", "English", "How much do I need to open an FDR, and what terms are available?", "The minimum is BDT 25,000. Tenures are 3, 6 or 12 months. Current rates: 7.50% for 3 months, 8.25% for 6 months, 9.00% for 12 months per year."),
    ("16", "Bangla", "FDR খুলতে ন্যূনতম কত টাকা লাগে এবং মেয়াদ কী কী?", "ন্যূনতম BDT 25,000। মেয়াদ ৩, ৬ বা ১২ মাস। বর্তমান বার্ষিক হার: ৩ মাসে 7.50%, ৬ মাসে 8.25% এবং ১২ মাসে 9.00%।"),
    ("16", "Banglish", "FDR khulte minimum koto taka lage, ar meyad koto?", "Minimum BDT 25,000. Meyad 3, 6 ba 12 mash. Current yearly rate: 3 mashe 7.50%, 6 mashe 8.25%, 12 mashe 9.00%."),
    ("17", "English", "What rate applies if I cash out my FDR before it matures?", "If you encash early, interest is paid at the savings account rate of 4% per year from the deposit date, and the FDR rate no longer applies."),
    ("17", "Bangla", "মেয়াদ শেষ হওয়ার আগে FDR ভাঙলে কী হারে সুদ পাব?", "আগে ভাঙালে জমার তারিখ থেকে সেভিংস অ্যাকাউন্টের বার্ষিক 4% হারে সুদ দেওয়া হবে; FDR-এর হার আর প্রযোজ্য হবে না।"),
    ("17", "Banglish", "Meyad shesh howar age FDR bhangle kon rate-e interest pabo?", "Age bhangle deposit-er date theke savings account-er yearly 4% rate-e interest paben; FDR rate ar projojjo hobe na."),
    ("18", "English", "What are the requirements for a personal loan?", "You must be 25 to 60 years old, salaried with a monthly income of at least BDT 30,000, and have at least 1 year in your current job. The maximum loan is BDT 10 lakh."),
    ("18", "Bangla", "পার্সোনাল লোন পেতে কী কী যোগ্যতা দরকার?", "আপনার বয়স ২৫ থেকে ৬০ বছরের মধ্যে হতে হবে, চাকরিজীবী হতে হবে, মাসিক আয় কমপক্ষে BDT 30,000 হতে হবে এবং বর্তমান চাকরিতে অন্তত ১ বছর থাকতে হবে। সর্বোচ্চ লোন BDT 10 লাখ।"),
    ("18", "Banglish", "Personal loan-er jonno ki ki joggota dorkar?", "Boyosh 25 theke 60 bochor hote hobe, salaried hote hobe, monthly income kompakhe BDT 30,000 hote hobe, ebong current job-e ontoto 1 bochor thakte hobe. Maximum loan BDT 10 lakh."),
    ("19", "English", "What is the yearly rate and processing charge for a personal loan?", "The interest rate is 12% per year on a reducing balance. There is a processing fee of 1% of the loan amount."),
    ("19", "Bangla", "পার্সোনাল লোনের বার্ষিক সুদের হার ও প্রসেসিং ফি কত?", "রিডিউসিং ব্যালেন্সে বার্ষিক সুদের হার 12%。 লোনের পরিমাণের 1% প্রসেসিং ফি আছে।"),
    ("19", "Banglish", "Personal loan-er yearly interest rate ar processing fee koto?", "Reducing balance-e yearly interest rate 12%. Loan amount-er 1% processing fee ache."),
    ("20", "English", "What portion of a home's value can the home loan cover?", "You can borrow up to 70% of the property value, with a repayment period of up to 20 years."),
    ("20", "Bangla", "বাড়ির দামের কত অংশ পর্যন্ত হোম লোন পাওয়া যায়?", "সম্পত্তির মূল্যের সর্বোচ্চ 70% পর্যন্ত লোন নিতে পারবেন। পরিশোধের মেয়াদ সর্বোচ্চ ২০ বছর।"),
    ("20", "Banglish", "Bashar property value-r koto percent porjonto home loan pawa jay?", "Property value-r maximum 70% porjonto loan nite parben. Repayment period maximum 20 bochor."),
    ("21", "English", "What salary and age do I need for a credit card?", "You must be 21 to 65 years old with a monthly income of at least BDT 40,000."),
    ("21", "Bangla", "ক্রেডিট কার্ড পেতে বয়স ও মাসিক আয়ের শর্ত কী?", "আপনার বয়স ২১ থেকে ৬৫ বছরের মধ্যে এবং মাসিক আয় কমপক্ষে BDT 40,000 হতে হবে।"),
    ("21", "Banglish", "Credit card pete boyosh ar monthly income-er condition ki?", "Boyosh 21 theke 65 bochor-er moddhe ebong monthly income kompakhe BDT 40,000 hote hobe."),
    ("22", "English", "On what dates is the credit card bill issued and due?", "Your statement is generated on the 5th of every month and the payment is due on the 25th of the same month."),
    ("22", "Bangla", "ক্রেডিট কার্ডের বিল কবে তৈরি হয় এবং কবে পরিশোধ করতে হয়?", "প্রতি মাসের ৫ তারিখে স্টেটমেন্ট তৈরি হয় এবং একই মাসের ২৫ তারিখে বিল পরিশোধ করতে হয়।"),
    ("22", "Banglish", "Credit card-er bill kokhon generate hoy ar kokhon pay korte hoy?", "Proti masher 5 tarikh-e statement hoy, ar oi masher 25 tarikh-e payment due thake."),
    ("23", "English", "What penalty do I pay for a late credit card payment?", "A flat late payment fee of BDT 500 applies if you miss the due date."),
    ("23", "Bangla", "ক্রেডিট কার্ডের বিল দেরিতে দিলে জরিমানা কত?", "নির্ধারিত তারিখে বিল না দিলে BDT 500 নির্দিষ্ট দেরি ফি প্রযোজ্য।"),
    ("23", "Banglish", "Credit card-er bill late dile fee koto?", "Due date miss korle flat late payment fee BDT 500."),
    ("24", "English", "How should I report a transaction I don't recognize?", "Report it within 30 days by calling 16999 or visiting your branch. We investigate within 15 working days and tell you the outcome."),
    ("24", "Bangla", "নিজে করিনি এমন লেনদেন দেখলে কীভাবে জানাব?", "৩০ দিনের মধ্যে 16999 নম্বরে কল করে বা শাখায় গিয়ে জানান। আমরা ১৫ কর্মদিবসের মধ্যে তদন্ত করে ফল জানাব।"),
    ("24", "Banglish", "Nijer kora noy emon transaction dekhle kibhabe report korbo?", "30 diner moddhe 16999-e call kore ba branch-e giye report korun. Amra 15 working day-er moddhe investigate kore result janabo."),
    ("25", "English", "What is the yearly price for SMS notifications?", "SMS alerts cost a flat BDT 120 per year, charged to your account."),
    ("25", "Bangla", "SMS অ্যালার্টের বার্ষিক খরচ কত?", "SMS অ্যালার্টের জন্য বছরে নির্দিষ্ট BDT 120 লাগে, যা আপনার অ্যাকাউন্ট থেকে কাটা হয়।"),
    ("25", "Banglish", "SMS alert-er yearly charge koto?", "SMS alert-er jonno bochore flat BDT 120 charge hoy, ja account theke kata hoy."),
    ("26", "English", "Do you charge anything to receive an overseas remittance?", "Receiving remittance is free. The money is credited within 24 hours of confirmation, to an account in your own name."),
    ("26", "Bangla", "বিদেশ থেকে রেমিট্যান্স গ্রহণে কোনো ফি লাগে?", "রেমিট্যান্স গ্রহণ বিনামূল্যে। নিশ্চিত হওয়ার ২৪ ঘণ্টার মধ্যে আপনার নিজের নামের অ্যাকাউন্টে টাকা জমা হয়।"),
    ("26", "Banglish", "Bidesh theke remittance receive korte kono fee lage?", "Remittance receive kora free. Confirmation-er 24 ghontar moddhe apnar nijer name-er account-e taka credit hoy."),
    ("27", "English", "How can I request a cheque book and how much does it cost?", "Request one at your branch or in the app. A 25-leaf cheque book costs BDT 250 and is ready in 3 working days."),
    ("27", "Bangla", "চেক বই কীভাবে চাইব এবং দাম কত?", "শাখায় বা অ্যাপে আবেদন করুন। ২৫ পাতার চেক বইয়ের দাম BDT 250 এবং ৩ কর্মদিবসে প্রস্তুত হয়।"),
    ("27", "Banglish", "Cheque book kibhabe request korbo, ar dam koto?", "Branch-e ba app-e request korun. 25-leaf cheque book-er dam BDT 250, ar 3 working day-e ready hoy."),
    ("28", "English", "How do I cancel payment on a cheque I've issued?", "Submit a written stop-payment request at your branch. The fee is BDT 100 per cheque."),
    ("28", "Bangla", "ইস্যু করা চেকের পেমেন্ট বন্ধ করতে কী করব?", "শাখায় লিখিত stop-payment অনুরোধ জমা দিন। প্রতি চেকের ফি BDT 100।"),
    ("28", "Banglish", "Issue kora cheque-er payment bondho korbo kibhabe?", "Branch-e written stop-payment request joma din. Proti cheque-er fee BDT 100."),
    ("29", "English", "What is the process and fee for shutting down my account?", "Visit your branch with your NID and fill in the closure form. If the account is closed within 6 months of opening, a fee of BDT 500 applies."),
    ("29", "Bangla", "অ্যাকাউন্ট বন্ধ করার নিয়ম ও ফি কী?", "NID নিয়ে শাখায় যান এবং বন্ধের ফর্ম পূরণ করুন। খোলার ৬ মাসের মধ্যে অ্যাকাউন্ট বন্ধ করলে BDT 500 ফি প্রযোজ্য।"),
    ("29", "Banglish", "Account bondho korar process ar fee ki?", "NID niye branch-e jan ebong closure form puron korun. Kholar 6 masher moddhe account bondho korle BDT 500 fee lage."),
    ("30", "English", "How frequently must I refresh my KYC documents?", "KYC must be updated every 5 years. We will send you an SMS reminder before the deadline. Bring your NID and a recent utility bill to your branch."),
    ("30", "Bangla", "কত বছর পরপর KYC তথ্য হালনাগাদ করতে হয়?", "প্রতি ৫ বছর পর KYC আপডেট করতে হবে। সময়সীমার আগে SMS মনে করিয়ে দেওয়া হবে। NID ও সাম্প্রতিক ইউটিলিটি বিল নিয়ে শাখায় আসুন।"),
    ("30", "Banglish", "Koto bochor por por KYC update korte hoy?", "Proti 5 bochor por KYC update korte hobe. Deadline-er age SMS reminder deya hobe. NID ebong recent utility bill niye branch-e asun."),
]


def build() -> int:
    with FAQ_PATH.open(encoding="utf-8-sig", newline="") as stream:
        faqs = {row["id"]: row for row in csv.DictReader(stream)}
    with QUESTIONS_PATH.open(encoding="utf-8-sig", newline="") as stream:
        test_questions = {row["question"].strip() for row in csv.DictReader(stream)}

    seen: set[tuple[str, str, str]] = set()
    rows = []
    for faq_id, language, question, answer in EXAMPLES:
        if faq_id not in faqs:
            raise ValueError(f"Training example refers to unknown FAQ ID {faq_id}.")
        if question.strip() in test_questions:
            raise ValueError(f"Evaluation question leaked into training data: {question}")
        key = (faq_id, language, question)
        if key in seen:
            raise ValueError(f"Duplicate training example: {key}")
        seen.add(key)
        faq = faqs[faq_id]
        context = f"FAQ ID: {faq_id}\nTopic: {faq['topic']}\nQuestion: {faq['question']}\nAnswer: {faq['answer']}"
        rows.append(
            {
                "faq_id": faq_id,
                "language": language,
                "question": question,
                "context": context,
                "answer": answer,
                "review_status": "reviewed",
            }
        )

    expected_ids = set(faqs)
    if {row["faq_id"] for row in rows} != expected_ids:
        raise ValueError("Training data must include examples for every FAQ.")
    if {row["language"] for row in rows} != {"English", "Bangla", "Banglish"}:
        raise ValueError("Training data must include all three supported languages.")

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} draft examples from {len(faqs)} FAQ entries to {OUTPUT_PATH}.")
    print("All examples are marked review_status=reviewed.")
    return len(rows)


if __name__ == "__main__":
    build()
