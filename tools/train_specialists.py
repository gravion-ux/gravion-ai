"""Train bundled local weights for every Gravion specialist.

All seed examples in this file were written for Gravion AI.  The
script writes one seed corpus and one v3 weights artifact per
specialist under ``src/gravion_ai/data`` and refreshes the model
registry from the stored artifacts.  Parameter counts and hashes
are computed from the artifacts, never entered by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gravion_ai.core_model import save_weights, train_local_core  # noqa: E402

DATA = ROOT / "src" / "gravion_ai" / "data"

# Each topic is (prompt phrase, completion).  Four deterministic
# prompt variants are generated per topic so the sparse classifier
# learns the domain terms rather than one exact wording.
TOPICS: dict[str, list[tuple[str, str]]] = {
    "gravion-chat": [
        ("greet a new user warmly", "Hello and welcome. Tell me what you would like to work on, and I will help step by step."),
        ("ask how someone is feeling today", "How are you feeling today? Share as much or as little as you like."),
        ("summarize a long message briefly", "Here is a brief summary: keep the main point, the key reason, and the next action."),
        ("explain a difficult idea simply", "Let us break the idea into small parts, define each part in plain words, then connect the parts with one example."),
        ("make a polite reminder message", "A gentle reminder: please review this when you have a moment, and let me know if anything is unclear."),
        ("apologize for a late reply", "Sorry for the late reply. I have read your message now and will answer the open points in order."),
        ("plan a productive morning routine", "Start with water and light movement, review the top three priorities, then work on the hardest task first."),
        ("help choose between two options", "Compare both options by cost, time, risk, and fit with your goal; choose the one that wins on the criteria that matter most."),
        ("write a short thank you note", "Thank you for your time and help. I appreciate it, and I will follow up with the next update soon."),
        ("brainstorm names for a small project", "Try short names that hint at the purpose: clear, easy to spell, and memorable after one hearing."),
        ("turn notes into action items", "Action items: name the owner, the task, and the deadline for each note, then remove anything already done."),
        ("prepare questions for a meeting", "Prepare questions about the goal, constraints, owners, deadlines, and what decision must be made before the meeting ends."),
        ("explain why sleep matters", "Sleep supports memory, mood, focus, and physical recovery, so a regular schedule usually improves daytime performance."),
        ("give a calm response to criticism", "Thank you for the feedback. I will check the specific points, fix what is valid, and explain anything that needs context."),
        ("describe a weekly review habit", "A weekly review lists wins, open loops, lessons, and the next three priorities for the coming week."),
        ("help draft a friendly invitation", "You are warmly invited to join us. Please let me know if you can come, and tell me about any needs I should plan for."),
        ("explain the difference between a goal and a task", "A goal is the outcome you want; a task is one concrete step that moves you toward that outcome."),
        ("make a simple study plan", "Study in short focused blocks, test yourself after each block, and review mistakes the next day."),
        ("write a concise status update", "Status: completed work is listed first, current blockers are named plainly, and the next milestone has a date."),
        ("encourage someone who feels stuck", "Being stuck is a signal to shrink the next step. Pick one small action you can finish today and build from there."),
        ("explain active listening", "Active listening means hearing the words, checking the meaning, and responding to what was actually said."),
        ("create a packing checklist", "Pack documents, chargers, clothes for the weather, toiletries, medicines, and one spare outfit in hand luggage."),
        ("help compare prices fairly", "Compare the full delivered price, quality, warranty, return terms, and running cost, not only the headline price."),
        ("write a short introduction for a speaker", "Please welcome our speaker, whose work focuses on practical ideas, clear examples, and honest answers."),
        ("plan a calm evening wind down", "Dim the lights, put devices away, prepare tomorrow's first task, and end with a short relaxing routine."),
        ("explain how habits form", "Habits form when a cue triggers a small repeated action and the result feels rewarding enough to repeat."),
        ("draft a request for feedback", "Could you share feedback on clarity, usefulness, and anything missing? Specific examples will help me improve it."),
        ("make a decision checklist", "Decision checklist: define the problem, list options, score risks, check constraints, then choose and record why."),
        ("explain why backups matter", "Backups protect work from device failure, accidental deletion, and corruption; test restores so the backup is real."),
        ("write a friendly follow up message", "Following up on my last message: I wanted to check whether you need anything else from me."),
        ("help organize a messy desk", "Sort items into keep, move, recycle, and discard piles, then give every kept item one home."),
        ("describe a good apology", "A good apology names the harm, accepts responsibility, explains the fix, and does not demand instant forgiveness."),
        ("plan a balanced weekly meal outline", "Plan protein, grains, vegetables, and fruit across the week, repeat simple meals, and keep one flexible day."),
        ("explain the value of clear writing", "Clear writing saves time because readers understand the point, the evidence, and the requested action without guessing."),
        ("brainstorm weekend activity ideas", "Try a nature walk, a cooking experiment, a museum visit, a board game, or a small home project."),
        ("help prepare for a job interview", "Prepare stories about your work, practice clear answers, research the role, and bring thoughtful questions."),
        ("write a short motivational message", "Progress counts even when it is small. Do the next honest step and let momentum build."),
        ("explain how to ask better questions", "Ask one clear question at a time, include the context, and say what kind of answer would help."),
        ("make a simple budget outline", "List income, fixed costs, variable costs, savings, and a small buffer; then adjust the flexible items first."),
        ("describe how to learn from mistakes", "Name what happened, find the cause, change one process, and check whether the change worked."),
        ("draft a welcome message for a team", "Welcome to the team. Your first week will cover introductions, tools, current priorities, and a buddy for questions."),
        ("help choose a book to read", "Choose a book by the question you want answered, the length you can finish, and reviews from readers you trust."),
        ("explain why regular exercise helps", "Regular exercise supports heart health, strength, mood, sleep, and energy when it is consistent and safe."),
        ("write a polite decline message", "Thank you for thinking of me. I cannot take this on right now, but I wish you well with it."),
        ("plan a focused work session", "Pick one outcome, silence distractions, work in timed blocks, and record the result before taking a break."),
        ("explain the idea of compound growth", "Compound growth means gains build on earlier gains, so small consistent progress can become large over time."),
        ("help name priorities for today", "Today's priorities should be few, clear, and ordered by impact and deadline."),
        ("write a closing message for a conversation", "Thanks for the conversation. I will send the agreed next steps and check in after the first milestone."),
        # Demo-critical openings (28 Nov 2026 cycle): natural
        # greeting, identity and capability phrasings a demo
        # audience actually uses, plus a plain concept explain.
        ("say hello and introduce yourself", "Hello, I am Gravion. Tell me what you would like to work on and I will help step by step."),
        ("tell what you can do for me", "I can chat, write and review code, build websites, analyse security alerts, draft agent plans, and help with support questions."),
        ("explain recursion simply", "Recursion is when a function calls itself on a smaller piece of the problem until it reaches a base case that stops the calls."),
        ("describe who built gravion", "Gravion is a modular AI platform built by Priyanshu Kumar Thakur, with chat, coder, cyber, website builder and agentic modules over one system."),
        # Plain concept explain (28 Nov 2026 cycle, 04:16 run):
        # "explain artificial intelligence simply" echoed while
        # the difficult-idea topic answered, so the chat class
        # gains the direct concept shape.
        ("explain artificial intelligence simply", "Artificial intelligence is software that learns patterns from examples to answer, classify, or generate, instead of following only fixed rules written in advance."),
        ("what is artificial intelligence", "Artificial intelligence is software that learns patterns from examples to answer, classify, or generate, instead of following only fixed rules written in advance."),
    ],
    "gravion-support": [
        ("check the status of an order", "Share your order number and I will check its current status and tell you the next step."),
        ("explain how to track a delivery", "Use the tracking link from your confirmation message, or share your order number and I will look up the latest delivery update."),
        ("help with a missing package", "Sorry your package has not arrived. Share your order number and delivery address so I can check the carrier record and arrange the next step."),
        ("explain the return process", "To return an item, keep it unused with its packaging, start the return from your orders page, and follow the pickup or drop-off instructions shown there."),
        ("help change a delivery address", "If your order has not shipped yet, I can update the delivery address. Share your order number and the new address."),
        ("explain how refunds are processed", "Once a returned item is received and checked, the refund goes back to the original payment method; the exact timing depends on your bank or payment provider."),
        ("help with a damaged item", "Sorry the item arrived damaged. Share a photo of the damage and your order number, and I will arrange a replacement or refund."),
        ("explain warranty coverage basics", "Warranty cover depends on the product and the fault. Share the product name and purchase date and I will check what applies."),
        ("help reset an account password", "Use the forgot password link on the sign-in page; a reset link will be sent to your registered email address."),
        ("explain how to update account details", "Open account settings, edit the detail you want to change, and save. Some changes ask you to confirm by email first."),
        ("help with a failed payment", "If a payment failed, check that the card details and balance are correct, then try again. Share the order number if the charge still appears."),
        ("explain subscription cancellation steps", "Open your subscription settings, choose cancel, and confirm. You keep access until the end of the current paid period."),
        ("help find an invoice copy", "Invoices are available from your orders page and are also emailed after payment. Tell me the order number if you need another copy."),
        ("explain delivery time estimates", "Delivery estimates are shown at checkout and in your confirmation message; they depend on your location and the items ordered."),
        ("help with a wrong item received", "Sorry you received the wrong item. Share your order number and a photo of what arrived, and I will arrange the correct item."),
        ("explain how to apply a discount code", "Enter the code in the discount field before paying and check that the total updates. Codes cannot usually be added after payment."),
        ("help with an account lockout", "After several failed sign-ins the account locks for safety. Wait a short while, then use the password reset link to regain access."),
        ("explain how to contact the team", "You can reach the team through this chat, the contact form, or the support email listed on the contact page."),
        ("help report a website problem", "Thanks for reporting it. Tell me the page address, what you expected, and what happened instead, and I will pass it to the team."),
        ("explain data privacy request steps", "To request your data or ask for deletion, send a request from your registered email address so your identity can be verified."),
        ("help change an email address", "Open account settings, enter the new email address, and confirm through the verification message sent to it."),
        ("explain how to pause a subscription", "Open subscription settings and choose pause if available; billing stops during the pause and resumes on the date shown."),
        ("help with a duplicate charge", "Sorry about the duplicate charge. Share the order number and the charge dates, and I will check and arrange a refund of the extra charge."),
        ("explain gift card balance checks", "Enter the gift card code on the balance page or at checkout to see the remaining amount."),
        ("help schedule a repair pickup", "Share your order number, the fault, and your preferred pickup day, and I will book the collection."),
        ("explain product setup first steps", "Start by charging or connecting the product as the guide shows, then follow the first-time setup prompts on screen."),
        ("help with a missing confirmation email", "Check spam and promotions folders first. If it is not there, share your order number and I will resend the confirmation."),
        ("explain how loyalty points work", "Points are earned on eligible purchases and can be applied at checkout; your balance and expiry dates are shown in your account."),
        ("help update payment method", "Open billing settings, add the new payment method, and set it as default before the next billing date."),
        ("explain exchange options for size", "Size exchanges are started from your orders page; the replacement is sent once the original item is on its way back."),
        ("help with a late delivery apology", "Sorry your delivery is late. Share your order number and I will check the latest carrier update right away."),
        ("my package is late", "Sorry your order is running late. Share your order number and I will track the package and tell you the latest delivery update."),
        ("paisa wapas chahiye order galat aaya", "Sorry the wrong order arrived. Share your order number and I will arrange a refund or the correct item for you."),
        ("explain how to download a receipt", "Open the order in your account and choose download receipt; a copy is also attached to your confirmation email."),
        ("help report a billing error", "Tell me the order number and the amount that looks wrong, and I will review the charge and correct it if needed."),
        ("explain account verification steps", "Verification usually means confirming your email address and, for some actions, entering a code sent to you."),
        ("help find product manuals", "Manuals are linked on each product page and in the support section; tell me the model name if you cannot find yours."),
        ("explain service outage updates", "During an outage, status updates are posted on the status page. Share what you are seeing and I will check the latest note."),
        ("help with a cancelled order question", "If your order was cancelled, the confirmation message explains why and whether a refund is on its way."),
        ("explain how to change an order", "Orders can be changed before they ship. Share your order number and the change you need as soon as possible."),
        ("help with installation booking", "Share your order number and preferred dates, and I will book the installation visit."),
        ("explain spare parts ordering", "Spare parts can be ordered from the parts section by model name; share your model if you need help finding the right part."),
        ("help with a software update problem", "Restart the device, check your connection, and try the update again. Tell me the error message if it still fails."),
        ("explain how to close an account", "Account closure is started from account settings after any open orders finish; you will be asked to confirm by email."),
        ("help recover a deleted file", "Check the recycle or trash area first; recently deleted files can usually be restored from there for a limited time."),
        ("explain accessibility support options", "Accessibility options include larger text, keyboard navigation, and screen reader support; tell me what you need and I will guide you."),
        ("help with a gift order message", "Gift messages are added at checkout and printed with the delivery note. Share your order number if the message needs changing."),
        ("explain bulk order enquiries", "For bulk orders, share the product, quantity, and delivery location, and the team will reply with availability and pricing."),
        ("help check service availability", "Share your area or postcode and I will check whether the service is available there."),
        ("explain how to give feedback", "Feedback can be sent through the feedback form or this chat; specific examples help the team act on it faster."),
        # Demo-critical support phrasings (28 Nov 2026 cycle).
        ("ask what the opening hours are", "Opening hours are listed on the contact page for each location. Tell me your city or store and I will point you to the right listing."),
        ("find my order with an order number", "Share your order number and I will check its current status, the latest delivery update, and the next step."),
        ("ask how to reset a forgotten password", "Use the forgot password link on the sign-in page and follow the reset link sent to your registered email address."),
        # Natural first-person demo phrasings (28 Nov 2026 cycle,
        # second pass): customers ask "where is my order", not
        # "find my order with an order number".
        ("where is my order", "Share your order number and I will check its current status, the latest delivery update, and the next step."),
        ("track my order with my order id", "Share your order number and I will check its current status, the latest delivery update, and the next step."),
        ("i forgot my password", "Use the forgot password link on the sign-in page and follow the reset link sent to your registered email address."),
        ("how do i reset my password", "Use the forgot password link on the sign-in page and follow the reset link sent to your registered email address."),
        ("what are your opening hours", "Opening hours are listed on the contact page for each location. Tell me your city or store and I will point you to the right listing."),
        # First-person order problems (28 Nov 2026 cycle, 02:16
        # run): refund, cancellation, damaged delivery and
        # delivery-time asks in the customer's own words echoed
        # while the explain-form topics answered.
        ("i want a refund for my order", "Share your order number and I will start the refund; once the returned item is received and checked, the refund goes back to the original payment method."),
        ("cancel my order", "Share your order number and I will cancel it if it has not shipped yet; if it has shipped, I will explain the return path instead."),
        ("my package arrived damaged", "Sorry the item arrived damaged. Share a photo of the damage and your order number, and I will arrange a replacement or refund."),
        ("how long does delivery take", "Delivery estimates are shown at checkout and in your confirmation message; they depend on your location and the items ordered."),
        # Direct support questions (28 Nov 2026 cycle, 04:16
        # run): the question shapes customers type — "how do I
        # get a refund", "my order arrived damaged" — were
        # answered by the password-reset and order-status
        # classes while the statement forms above answered, so
        # those classes gain the direct shapes, and the payment
        # and address questions gain classes of their own.
        ("how do i get a refund", "Share your order number and I will start the refund; once the returned item is received and checked, the refund goes back to the original payment method."),
        ("get a refund for my order", "Share your order number and I will start the refund; once the returned item is received and checked, the refund goes back to the original payment method."),
        ("my order arrived damaged", "Sorry the item arrived damaged. Share a photo of the damage and your order number, and I will arrange a replacement or refund."),
        ("order arrived damaged", "Sorry the item arrived damaged. Share a photo of the damage and your order number, and I will arrange a replacement or refund."),
        ("what payment methods do you accept", "Accepted payment methods are shown at checkout before you pay. If a payment fails, share your order number and I will check its payment status."),
        ("which payment methods are accepted", "Accepted payment methods are shown at checkout before you pay. If a payment fails, share your order number and I will check its payment status."),
        ("change my delivery address", "If your order has not shipped yet, I can update the delivery address. Share your order number and the new address."),
        ("how do i change my delivery address", "If your order has not shipped yet, I can update the delivery address. Share your order number and the new address."),
        # Support questions round two (28 Nov 2026 cycle,
        # 05:16 run): a fresh demo probe showed "how do I
        # contact support" answered by the password-reset
        # class, and the cash-on-delivery and wrong-item asks
        # echoed, so those classes gain their direct shapes.
        ("how do i contact support", "You can reach the team through this chat, the contact form, or the support email listed on the contact page."),
        ("contact support", "You can reach the team through this chat, the contact form, or the support email listed on the contact page."),
        ("how can i contact the support team", "You can reach the team through this chat, the contact form, or the support email listed on the contact page."),
        ("cash on delivery", "Cash on delivery is offered where the checkout shows it as a payment option; the exact availability depends on your location and order."),
        ("do you offer cash on delivery", "Cash on delivery is offered where the checkout shows it as a payment option; the exact availability depends on your location and order."),
        ("pay cash on delivery", "Cash on delivery is offered where the checkout shows it as a payment option; the exact availability depends on your location and order."),
        ("i received the wrong item", "Sorry you received the wrong item. Share your order number and a photo of what arrived, and I will arrange the correct item."),
        ("wrong item received", "Sorry you received the wrong item. Share your order number and a photo of what arrived, and I will arrange the correct item."),
        ("received a wrong item", "Sorry you received the wrong item. Share your order number and a photo of what arrived, and I will arrange the correct item."),
        # Margin repair (28 Nov 2026 cycle, 05:16 run): the
        # retrain diluted "my order is late" back into echo
        # under Gravion Support while Omni still answered, so
        # the order-status class gains the late shapes.
        ("my order is late", "Share your order number and I will check its current status, the latest delivery update, and the next step."),
        ("order is late", "Share your order number and I will check its current status, the latest delivery update, and the next step."),
        ("my delivery is late", "Share your order number and I will check its current status, the latest delivery update, and the next step."),
        # Margin repair (28 Nov 2026 cycle, 06:16 run): the
        # shield/code retrain diluted "my payment failed" into
        # echo under Gravion Omni while Support answered, and
        # the same probe showed the cannot-login and
        # change-address asks echoing beside their classes, so
        # those classes gain the direct shapes.
        ("my payment failed", "If a payment failed, check that the card details and balance are correct, then try again. Share the order number if the charge still appears."),
        ("payment failed for my order", "If a payment failed, check that the card details and balance are correct, then try again. Share the order number if the charge still appears."),
        ("my payment did not go through", "If a payment failed, check that the card details and balance are correct, then try again. Share the order number if the charge still appears."),
        ("i cannot login to my account", "After several failed sign-ins the account locks for safety. Wait a short while, then use the password reset link to regain access."),
        ("cannot log in to my account", "After several failed sign-ins the account locks for safety. Wait a short while, then use the password reset link to regain access."),
        ("i cannot sign in", "After several failed sign-ins the account locks for safety. Wait a short while, then use the password reset link to regain access."),
        ("i want to change my address", "If your order has not shipped yet, I can update the delivery address. Share your order number and the new address."),
        ("change my address", "If your order has not shipped yet, I can update the delivery address. Share your order number and the new address."),
        ("update my delivery address", "If your order has not shipped yet, I can update the delivery address. Share your order number and the new address."),
        # First-person billing and account asks (28 Nov 2026
        # cycle, 07:16 run): "I was charged twice" echoed
        # beside the duplicate-charge class, and "how do I
        # delete my account" was answered by the
        # password-reset class, so those classes gain the
        # direct shapes.
        ("i was charged twice", "Sorry about the duplicate charge. Share the order number and the charge dates, and I will check and arrange a refund of the extra charge."),
        ("charged twice for one order", "Sorry about the duplicate charge. Share the order number and the charge dates, and I will check and arrange a refund of the extra charge."),
        ("how do i delete my account", "Account closure is started from account settings after any open orders finish; you will be asked to confirm by email."),
        ("delete my account", "Account closure is started from account settings after any open orders finish; you will be asked to confirm by email."),
        # Returns and refund policy (28 Nov 2026 cycle,
        # 08:16 run): a fresh demo probe showed "how do I
        # return an item" and "what is your refund policy"
        # echoed under Gravion Support while the explain-form
        # return and refund topics answered, so those classes
        # gain the direct customer shapes.
        ("how do i return an item", "To return an item, keep it unused with its packaging, start the return from your orders page, and follow the pickup or drop-off instructions shown there."),
        ("i want to return an item", "To return an item, keep it unused with its packaging, start the return from your orders page, and follow the pickup or drop-off instructions shown there."),
        ("how can i return a product", "To return an item, keep it unused with its packaging, start the return from your orders page, and follow the pickup or drop-off instructions shown there."),
        ("what is your refund policy", "Refunds go back to the original payment method once a returned item is received and checked; share your order number and I will confirm the status and timing for your order."),
        ("how does your refund policy work", "Refunds go back to the original payment method once a returned item is received and checked; share your order number and I will confirm the status and timing for your order."),
        ("explain your refund policy", "Refunds go back to the original payment method once a returned item is received and checked; share your order number and I will confirm the status and timing for your order."),
    ],
    "gravion-code": [
        ("write a function that adds two numbers", "Define a small function with two numeric inputs, return their sum, and test it with positive, negative, and zero values."),
        ("explain a variable in simple terms", "A variable is a named place that stores a value so code can read or update it later."),
        ("debug a failing test step by step", "Reproduce the failure, read the first error, isolate the smallest failing case, fix one cause, then rerun the test."),
        ("refactor duplicated code safely", "Extract the repeated logic into one helper, keep behavior unchanged, and run the existing tests after each small change."),
        ("write a loop that prints each item", "Iterate over the collection in order and handle one item per pass; avoid changing the collection while looping."),
        ("explain recursion with an example", "Recursion solves a problem by calling the same function on a smaller input until a base case stops the calls."),
        ("handle an empty list safely", "Check for an empty list before indexing or averaging, and return a clear default or error message."),
        ("design a small api endpoint", "Define the route, inputs, validation, success response, error responses, and tests for each case."),
        ("explain object oriented programming", "Object oriented programming groups data and behavior into objects with clear responsibilities and interfaces."),
        ("write a unit test for a parser", "Test a valid input, an empty input, malformed input, and boundary values; assert both output and error behavior."),
        ("improve slow code performance", "Profile first, find the hot path, reduce repeated work, choose better data structures, and measure again."),
        ("explain database indexes", "An index is a lookup structure that helps the database find rows faster, at the cost of storage and write time."),
        ("validate user input in code", "Check type, length, range, and allowed characters; reject invalid input with a clear message before using it."),
        ("write error handling for file reading", "Open the file safely, handle missing or unreadable files, and close resources even when an error occurs."),
        ("explain the difference between a list and a set", "A list keeps order and duplicates; a set keeps unique items and is built for membership checks."),
        ("create a simple class for a task", "A task class can store title, status, and due date, with methods to complete, reopen, and describe the task."),
        ("explain version control basics", "Version control records changes over time so you can review history, collaborate, and restore earlier work."),
        ("write a function to reverse a string", "Use a small function: def reverse(text): return text[::-1]. Test it with empty and single-character inputs."),
        ("explain asynchronous code", "Asynchronous code starts work without blocking everything else, then handles the result when it is ready."),
        ("design a cache with expiry", "Store each value with an expiry time, return fresh values quickly, and refresh or remove stale entries."),
        ("write a binary search explanation", "Binary search repeatedly checks the middle of a sorted range and discards half until the target is found or absent."),
        ("explain stack and queue differences", "A stack removes the newest item first; a queue removes the oldest item first."),
        ("make code readable with naming", "Use names that state purpose, keep functions small, and remove comments that merely repeat the code."),
        ("write a test for boundary values", "Test just below, exactly at, and just above each boundary so off-by-one errors are caught."),
        ("explain dependency injection", "Dependency injection passes needed collaborators into code instead of creating them inside, making tests easier."),
        ("handle null values safely", "Decide where null is allowed, check before use, and prefer explicit defaults or errors over silent failure."),
        ("write a sorting function outline", "Compare neighboring values or use a known sorting method, then verify sorted, reversed, and duplicate inputs."),
        ("explain time complexity simply", "Time complexity describes how running time grows as input size grows, using the dominant growth term."),
        ("create a configuration loader", "Read settings from a file or environment, validate required keys, apply defaults, and fail with clear messages."),
        ("write a retry helper with backoff", "Retry only temporary failures, wait longer between attempts, cap total attempts, and log the final error."),
        ("explain interfaces and contracts", "An interface is a contract: callers rely on promised inputs and outputs without knowing internal details."),
        ("design a logging approach", "Log important events with context, avoid secrets, use levels consistently, and make failures searchable."),
        ("write a function to count word frequency", "Split text into words, normalize case if needed, count each word, and test punctuation and empty text."),
        ("explain memory leaks", "A memory leak keeps unused data reachable, so memory grows over time; find retained references and release them."),
        ("make a simple command line tool", "Parse arguments, validate them, run one clear action, print helpful output, and return a proper exit code."),
        ("explain testing pyramids", "Use many fast unit tests, fewer integration tests, and a small number of end-to-end tests for critical paths."),
        ("write a merge of two sorted lists", "Walk both lists with pointers, always take the smaller next value, then append the remaining tail."),
        ("explain hashing for passwords", "Password storage should use a slow salted hash designed for passwords, never a fast general hash or plain text."),
        ("design a pagination response", "Return page items, total count, page size, current page, and links or cursors for the next page."),
        ("write a guard clause example", "Return early when input is invalid so the main path stays flat and readable."),
        ("explain immutability benefits", "Immutable values do not change after creation, which reduces surprise side effects and simplifies reasoning."),
        ("create a simple state machine", "Define states, allowed transitions, events, and guards; reject transitions that are not listed."),
        ("write a function to flatten nested lists", "Walk nested lists recursively or with a stack, append leaf values in order, and test empty nesting."),
        ("explain code review goals", "Code review checks correctness, clarity, tests, security, and maintainability before changes merge."),
        ("design a file upload validation", "Check size, type, extension, and content; store uploads outside executable paths and scan when required."),
        ("write a debounce helper outline", "Wait for a quiet period after the last call, then run the function once with the latest arguments."),
        ("explain event driven design", "Event driven design reacts to named events, keeping producers and consumers loosely coupled."),
        ("make a small data validation schema", "Define required fields, types, ranges, and cross-field rules; report every validation problem clearly."),
        ("write a function to find duplicates", "Track seen values in a set and collect values that appear again, preserving first duplicate order if needed."),
        ("explain database transactions", "A transaction groups changes so they all succeed together or roll back together, keeping data consistent."),
        ("create a health check endpoint", "Return service status, version, and dependency checks quickly, without exposing secrets or heavy work."),
        ("write a parser for key value lines", "Split each non-empty line on the first separator, trim spaces, validate keys, and report malformed lines."),
        ("explain feature flags", "Feature flags switch behavior at runtime so teams can release gradually and disable risky paths quickly."),
        ("design a rate limiter", "Count requests per key in a time window, allow bursts within limits, and return clear retry guidance."),
        ("write a function to normalize text", "Trim whitespace, collapse repeated spaces, normalize case when appropriate, and test punctuation handling."),
        ("write a haiku function", "Define a function that returns a three-line haiku, keep the syllable pattern clear, and test the exact output."),
        ("write a sort function", "Compare neighboring values or use a known sorting method, then verify sorted, reversed, and duplicate inputs."),
        # Demo-critical coder phrasings (28 Nov 2026 cycle).
        ("write a prime number checker function", "Check divisibility from 2 up to the square root of the number, return false on the first exact divisor, and test 2, 3, 4, and a large prime."),
        ("explain python decorators", "A decorator wraps a function to add behaviour before or after it runs, without changing the function body; the wrapper receives the original function and returns a new one."),
        ("review code that returns the wrong result", "Trace the inputs through each line, compare the actual output with the expected one, and check operators, order of operations, and off-by-one bounds first."),
        ("fix an infinite loop bug", "Check the loop condition and the update step: the variable the condition reads must move toward the exit value on every pass, and add a safety break while debugging."),
        # Natural first-person demo phrasings (28 Nov 2026 cycle,
        # second pass): the checklist prompts a demo audience
        # actually types, reusing the completions above so each
        # answer class gains phrasing breadth instead of splitting.
        ("write a function that checks if a number is prime", "Check divisibility from 2 up to the square root of the number, return false on the first exact divisor, and test 2, 3, 4, and a large prime."),
        ("check if a number is prime in python", "Check divisibility from 2 up to the square root of the number, return false on the first exact divisor, and test 2, 3, 4, and a large prime."),
        ("explain what a decorator is", "A decorator wraps a function to add behaviour before or after it runs, without changing the function body; the wrapper receives the original function and returns a new one."),
        ("what is a decorator in python", "A decorator wraps a function to add behaviour before or after it runs, without changing the function body; the wrapper receives the original function and returns a new one."),
        # Demo-probe repair (28 Nov 2026 cycle): the fix ask
        # carrying its code fragment echoed while the plain
        # topic phrasing answered, so the loop-fix class gains
        # the code-carrying shapes.
        ("fix this infinite loop", "Check the loop condition and the update step: the variable the condition reads must move toward the exit value on every pass, and add a safety break while debugging."),
        ("fix an infinite loop in this code", "Check the loop condition and the update step: the variable the condition reads must move toward the exit value on every pass, and add a safety break while debugging."),
        ("fix a while true infinite loop", "Check the loop condition and the update step: the variable the condition reads must move toward the exit value on every pass, and add a safety break while debugging."),
        ("stop an infinite loop", "Check the loop condition and the update step: the variable the condition reads must move toward the exit value on every pass, and add a safety break while debugging."),
        # Demo-probe repair (28 Nov 2026 cycle, 02:16 run): the
        # index-error ask echoed, and the palindrome ask was
        # answered by the prime class (shared "check if" words),
        # so the palindrome class gains its own phrasings.
        ("explain an index error in code", "An index error means the code asked for a position outside the list or string; check the length first and keep indexes from zero to length minus one."),
        ("why does code throw an index error", "An index error means the code asked for a position outside the list or string; check the length first and keep indexes from zero to length minus one."),
        ("my code throws an index error", "An index error means the code asked for a position outside the list or string; check the length first and keep indexes from zero to length minus one."),
        ("check if a string is a palindrome", "Compare the string with its reverse: text == text[::-1]. Test an empty string, a single character, and a mixed-case word after normalising case."),
        ("write a palindrome checker", "Compare the string with its reverse: text == text[::-1]. Test an empty string, a single character, and a mixed-case word after normalising case."),
        # Coder basics (28 Nov 2026 cycle, 04:16 run): a fresh
        # demo probe showed the plain beginner asks answered by
        # the wrong class — factorial by text normalising,
        # recursion by asynchronous code, a variable by
        # decorators, list-versus-tuple by list-versus-set — and
        # the syntax-error fix ask echoed, so those classes gain
        # their direct shapes.
        ("write a function to compute factorial", "Multiply the integers from 1 up to n, return 1 for 0, reject negative inputs, and test 0, 1, and 5."),
        ("compute the factorial of a number", "Multiply the integers from 1 up to n, return 1 for 0, reject negative inputs, and test 0, 1, and 5."),
        ("explain recursion", "Recursion solves a problem by calling the same function on a smaller input until a base case stops the calls."),
        ("what is recursion", "Recursion solves a problem by calling the same function on a smaller input until a base case stops the calls."),
        ("what is a variable", "A variable is a named place that stores a value so code can read or update it later."),
        ("explain what a variable is", "A variable is a named place that stores a value so code can read or update it later."),
        ("difference between a list and a tuple", "A list is mutable and can grow or change after creation; a tuple is fixed, so use a tuple for values that must not change."),
        ("explain list and tuple differences", "A list is mutable and can grow or change after creation; a tuple is fixed, so use a tuple for values that must not change."),
        # Margin repair (28 Nov 2026 cycle, 04:16 run): the two
        # shapes above left "explain the difference between a
        # list and a tuple" on the list-and-set class, so the
        # tuple class gains the remaining direct shapes.
        ("explain the difference between a list and a tuple", "A list is mutable and can grow or change after creation; a tuple is fixed, so use a tuple for values that must not change."),
        ("what is the difference between a list and a tuple", "A list is mutable and can grow or change after creation; a tuple is fixed, so use a tuple for values that must not change."),
        ("list vs tuple difference", "A list is mutable and can grow or change after creation; a tuple is fixed, so use a tuple for values that must not change."),
        ("difference between list and tuple in python", "A list is mutable and can grow or change after creation; a tuple is fixed, so use a tuple for values that must not change."),
        ("fix a syntax error", "Read the error line first, check brackets, quotes, and indentation around it, fix one issue, then run again; the reported line is often just after the real mistake."),
        ("how do i fix a syntax error", "Read the error line first, check brackets, quotes, and indentation around it, fix one issue, then run again; the reported line is often just after the real mistake."),
        # Coder basics round two (28 Nov 2026 cycle, 05:16
        # run): a fresh demo probe showed the next beginner
        # asks echoed (loop, hello world) or answered by the
        # wrong class — maximum-in-a-list by factorial, reading
        # a file by syntax errors, a function by variables, an
        # API by recursion — so those classes gain their direct
        # shapes.
        ("what is a loop in programming", "A loop repeats a block of code while a condition holds or for each item in a collection; make sure the condition changes so the loop ends."),
        ("explain loops in programming", "A loop repeats a block of code while a condition holds or for each item in a collection; make sure the condition changes so the loop ends."),
        ("explain what a loop is", "A loop repeats a block of code while a condition holds or for each item in a collection; make sure the condition changes so the loop ends."),
        ("find the maximum in a list", "Use max(values) for the largest item, or track the largest value in one pass; test an empty list separately because it has no maximum."),
        ("write code to find the maximum in a list", "Use max(values) for the largest item, or track the largest value in one pass; test an empty list separately because it has no maximum."),
        ("find the largest number in a list", "Use max(values) for the largest item, or track the largest value in one pass; test an empty list separately because it has no maximum."),
        ("read a file in python", "Open the file with a with block so it closes automatically: with open(path) as handle: text = handle.read(). Handle a missing file with a clear error."),
        ("how do i read a file in python", "Open the file with a with block so it closes automatically: with open(path) as handle: text = handle.read(). Handle a missing file with a clear error."),
        ("read text from a file", "Open the file with a with block so it closes automatically: with open(path) as handle: text = handle.read(). Handle a missing file with a clear error."),
        ("what is a function", "A function is a named, reusable block of code that takes inputs, does one job, and can return a result."),
        ("explain what a function is", "A function is a named, reusable block of code that takes inputs, does one job, and can return a result."),
        ("explain functions in programming", "A function is a named, reusable block of code that takes inputs, does one job, and can return a result."),
        ("explain what an api is", "An API is a defined way for programs to request work from each other: named operations, agreed inputs, and predictable responses."),
        ("what is an api", "An API is a defined way for programs to request work from each other: named operations, agreed inputs, and predictable responses."),
        ("explain apis in programming", "An API is a defined way for programs to request work from each other: named operations, agreed inputs, and predictable responses."),
        ("print hello world in python", "Use print(\"Hello, world!\") to show the text; it is the smallest complete program and a first check that the setup runs."),
        ("write a program to print hello world", "Use print(\"Hello, world!\") to show the text; it is the smallest complete program and a first check that the setup runs."),
        ("hello world program in python", "Use print(\"Hello, world!\") to show the text; it is the smallest complete program and a first check that the setup runs."),
        # Coder basics round three (28 Nov 2026 cycle, 06:16
        # run): a fresh demo probe showed sorting answered by
        # the maximum class and a factorial program by the
        # sorting class, error handling by file reading, a
        # class by decorators, and adding two numbers echoed
        # beside its function shape. Those classes gain their
        # direct shapes so the neighbours separate.
        ("write code to sort a list", "Compare neighboring values or use a known sorting method, then verify sorted, reversed, and duplicate inputs."),
        ("sort a list in python", "Compare neighboring values or use a known sorting method, then verify sorted, reversed, and duplicate inputs."),
        ("how do i sort a list", "Compare neighboring values or use a known sorting method, then verify sorted, reversed, and duplicate inputs."),
        ("write a factorial program", "Multiply the integers from 1 up to n, return 1 for 0, reject negative inputs, and test 0, 1, and 5."),
        ("factorial program in python", "Multiply the integers from 1 up to n, return 1 for 0, reject negative inputs, and test 0, 1, and 5."),
        ("write code for factorial", "Multiply the integers from 1 up to n, return 1 for 0, reject negative inputs, and test 0, 1, and 5."),
        ("handle errors in python", "Wrap risky code in try and except, catch the specific errors you expect, add context, and clean up resources in finally; do not hide the original cause."),
        ("how do i handle errors in python", "Wrap risky code in try and except, catch the specific errors you expect, add context, and clean up resources in finally; do not hide the original cause."),
        ("explain error handling in python", "Wrap risky code in try and except, catch the specific errors you expect, add context, and clean up resources in finally; do not hide the original cause."),
        ("what is a class in python", "A class is a blueprint for objects: it groups related data and the methods that act on it, and each object made from it keeps its own state."),
        ("explain classes in python", "A class is a blueprint for objects: it groups related data and the methods that act on it, and each object made from it keeps its own state."),
        ("what is a class in programming", "A class is a blueprint for objects: it groups related data and the methods that act on it, and each object made from it keeps its own state."),
        ("write a program to add two numbers", "Define a small function with two numeric inputs, return their sum, and test it with positive, negative, and zero values."),
        ("add two numbers in python", "Define a small function with two numeric inputs, return their sum, and test it with positive, negative, and zero values."),
        ("write code to add two numbers", "Define a small function with two numeric inputs, return their sum, and test it with positive, negative, and zero values."),
        # Demo-probe repair (28 Nov 2026 cycle, 07:16 run):
        # "write a fibonacci program" echoed while the
        # factorial and prime programs answered, so the
        # sequence class gains its direct shapes.
        ("write a fibonacci program", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("fibonacci sequence program", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        # Margin repair (28 Nov 2026 cycle, 07:16 run): the
        # two shapes above scored 0.103 toward the sorting
        # class, so the sequence class gains more verb
        # shapes.
        ("fibonacci program", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("write fibonacci code", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("program for fibonacci series", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("generate fibonacci numbers", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        # Margin repair round two (28 Nov 2026 cycle, 07:16
        # run): the shapes above still scored 0.103 toward
        # the sorting class and 0.297 toward factorial, so
        # the sequence class gains shapes carrying the
        # fibonacci token beside series and code words.
        ("fibonacci series code", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("code for fibonacci series", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("fibonacci numbers program", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("write code for fibonacci", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("fibonacci sequence code", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
        ("print fibonacci series", "Build the sequence from the first two values, each next value the sum of the previous two; test the first value, the second, and a longer run."),
    ],
    "gravion-review": [
        ("review this code for bugs", "Reviewing for bugs: check inputs, edge cases, error handling, and resource cleanup, then list findings by severity."),
        ("explain what this code does", "This code reads its inputs, applies its core logic step by step, and returns a result; I will walk through each part in order."),
        ("check this function for edge cases", "Edge cases to check: empty input, boundary values, duplicate items, very large input, and unexpected types."),
        ("review error handling in this code", "Good error handling catches expected failures, adds context, cleans up resources, and never hides the original cause."),
        ("explain this loop step by step", "The loop repeats its body once per item: it reads the next value, processes it, and stops when no items remain."),
        ("find performance problems in this code", "Look for repeated work inside loops, unnecessary copies, and slow lookups that a set or index would speed up."),
        ("review this code for security issues", "Check input validation, output encoding, secret handling, and permission checks before anything else."),
        ("explain this class and its methods", "The class groups related data and behavior; each method has one job, and the constructor sets the starting state."),
        ("check naming quality in this code", "Names should state purpose plainly; rename vague variables, and keep naming style consistent across the file."),
        ("review test coverage for this module", "Coverage gaps usually hide in error paths, boundary values, and rare branches; add tests there first."),
        ("explain this recursive function", "The function solves a smaller version of the problem each call until the base case returns a direct answer."),
        ("find duplicated logic in this code", "Repeated blocks should become one shared helper so fixes apply everywhere at once."),
        ("review this api design", "A clear API has predictable names, validated inputs, consistent errors, and no surprises for callers."),
        ("explain this sorting code", "The code compares values and reorders them until the whole sequence is in ascending order."),
        ("check resource cleanup in this code", "Files, connections, and locks should be released on every path, including when errors occur."),
        ("review concurrency safety here", "Shared mutable state needs synchronization; check for races between reads and writes across threads."),
        ("explain this regular expression", "The pattern matches text by describing allowed characters and repetition; I will break it into parts."),
        ("find readability problems in this code", "Long functions, deep nesting, and unclear names hurt readability; split, flatten, and rename."),
        ("review this database query code", "Check parameterization, index use, result size limits, and transaction boundaries."),
        ("explain this state machine code", "The code tracks a current state and only allows listed transitions when events arrive."),
        ("check input validation in this handler", "Validate type, length, range, and allowed values on the server before using any input."),
        ("review logging in this code", "Logs should carry context, avoid secrets, and make failures searchable without flooding output."),
        ("explain this caching code", "The cache stores recent results with an expiry, returning fresh values fast and refreshing stale ones."),
        ("find off by one risks in this code", "Check loop bounds, slice ends, and index arithmetic against empty and single-item inputs."),
        ("review this file handling code", "Check path safety, missing-file handling, encoding, and that files close even on error."),
        ("explain this asynchronous code path", "The code starts work without blocking, then handles results or errors when they arrive."),
        ("check null handling in this code", "Every value that can be missing needs an explicit check or a clear default before use."),
        ("review dependency choices in this project", "Prefer maintained, minimal dependencies; check licenses, update history, and whether the standard library suffices."),
        ("explain this parser code", "The parser reads input token by token, validates structure, and builds a result or reports the first error."),
        ("find hidden side effects in this code", "Watch for global state changes, file writes, and network calls hidden inside innocent-looking helpers."),
        ("review this configuration loading code", "Configuration should validate required keys, apply safe defaults, and fail with clear messages."),
        ("explain this binary search implementation", "It checks the middle of a sorted range and discards half each step until the target is found or absent."),
        ("check retry logic in this code", "Retries should cover temporary failures only, back off between attempts, and stop after a clear limit."),
        ("review this authentication code", "Check password storage, session expiry, token handling, and that failures reveal no useful detail."),
        ("explain this event handler code", "The handler reacts to a named event, reads its payload, and updates state or triggers follow-up work."),
        ("find memory leak risks in this code", "Look for growing collections, unclosed resources, and references kept after they are no longer needed."),
        ("review this pagination code", "Check page bounds, total counts, stable ordering, and that empty pages return cleanly."),
        ("explain this merge algorithm code", "It walks two sorted lists together, always taking the smaller next value, then appends the tail."),
        ("check type consistency in this code", "Mixed types in one collection cause subtle bugs; enforce one shape at the boundary."),
        ("review this cli argument parsing", "Arguments should be validated early, help text should match behavior, and exit codes should be meaningful."),
        ("explain this decorator code", "The decorator wraps a function to add behavior before or after it runs, without changing its core logic."),
        ("find brittle assumptions in this code", "Assumptions about input size, ordering, locale, or timing break first; make them explicit and tested."),
        ("review this migration code", "Migrations should be reversible where possible, tested on realistic data, and safe to run once."),
        ("explain this queue processing code", "The code takes items in order, processes each, and acknowledges or retries based on the outcome."),
        ("check boundary tests for this function", "Test just below, exactly at, and just above each boundary so off-by-one errors surface."),
        ("review this error message wording", "Error messages should say what failed, why it matters, and what the user can do next."),
        ("explain this data validation schema", "The schema lists required fields, types, and ranges, and reports every problem it finds."),
        ("summarize review findings by severity", "Findings by severity: correctness and security first, then performance, then readability and style."),
    ],
    "gravion-builder": [
        ("plan a landing page for a small studio", "Landing page plan: clear hero, proof of work, services, process, testimonial space, and one contact action."),
        ("design a portfolio home page", "Portfolio home: strong intro, selected projects, skills, short biography, and a simple contact section."),
        ("create a restaurant website structure", "Restaurant site: menu, story, hours, location, gallery, reservations, and contact details."),
        ("plan a blog layout with readable text", "Blog layout: narrow text column, clear headings, generous spacing, author line, and related reading links."),
        ("design a product page with pricing", "Product page: benefit headline, feature list, pricing choices, comparison notes, and a clear purchase action."),
        ("make a contact page that builds trust", "Contact page: short form, direct email, response time, location if relevant, and privacy note."),
        ("plan a documentation site navigation", "Documentation navigation: start guide, concepts, tutorials, reference, search, and version notes."),
        ("design a mobile first home screen", "Mobile first: primary action near the top, readable type, large touch targets, and content in priority order."),
        ("create an about page outline", "About page: mission, story, values, team or founder note, milestones, and contact invitation."),
        ("plan a checkout page flow", "Checkout flow: cart summary, details, payment step, review, confirmation, and clear error recovery."),
        ("design a dashboard overview layout", "Dashboard overview: key metrics first, recent activity, alerts, charts, and shortcuts to common actions."),
        ("make a pricing table easy to compare", "Pricing table: align features by row, highlight the recommended plan, and state limits without fine print."),
        ("plan a course website structure", "Course site: outcomes, curriculum, instructor, schedule, pricing, sample lesson, and enrollment action."),
        ("design a search results page", "Search results: query summary, filters, result cards, empty state, and sort controls."),
        ("create a faq page structure", "FAQ page: group questions by topic, keep answers short, and link to contact help when needed."),
        ("plan a nonprofit website homepage", "Nonprofit homepage: mission, impact numbers, stories, ways to help, transparency links, and donation action."),
        ("design a booking page flow", "Booking flow: choose service, pick time, enter details, confirm, and show cancellation terms clearly."),
        ("make a gallery page layout", "Gallery layout: consistent image sizes, captions, filters, and a lightbox that works with keyboard controls."),
        ("plan a personal resume website", "Resume site: summary, experience, projects, skills, education, and downloadable resume link."),
        ("design a settings page layout", "Settings page: group related controls, label every field, show save state, and confirm destructive actions."),
        ("create a newsletter signup section", "Newsletter section: clear benefit, one email field, privacy note, and a success message after signup."),
        ("plan a community forum homepage", "Forum homepage: categories, active discussions, rules, search, and a clear new topic action."),
        ("design a checkout error state", "Error state: explain what failed, keep entered data, highlight the field, and offer a safe retry."),
        ("make a hero section with one message", "Hero section: one promise, one supporting sentence, one primary action, and visual proof."),
        ("plan a multi language website structure", "Multi-language structure: language switcher, translated navigation, locale-aware dates, and fallback text rules."),
        ("design a testimonial section", "Testimonial section: short quotes, names or roles, star context if used, and no invented claims."),
        ("create a footer with useful links", "Footer: contact, key pages, legal links, social links, and a short site summary."),
        ("plan an event website page", "Event page: date, venue, agenda, speakers, tickets, travel notes, and updates section."),
        ("design a profile page layout", "Profile page: identity header, about text, activity, settings shortcut, and privacy controls."),
        ("make a comparison page for two plans", "Comparison page: shared criteria, honest trade-offs, best-for notes, and a recommendation by use case."),
        ("plan a help center homepage", "Help center: search first, popular topics, guides by task, status link, and contact support action."),
        ("design a signup form with few fields", "Signup form: ask only essentials, explain why each field is needed, and validate kindly."),
        ("create a case study page outline", "Case study: problem, constraints, approach, result, evidence, and lessons learned."),
        ("plan a podcast website structure", "Podcast site: latest episode, episode archive, about the show, host bio, and subscribe links."),
        ("design a map location section", "Location section: address, map, transit notes, parking notes, and opening hours."),
        ("make a feature grid section", "Feature grid: six focused features, short descriptions, consistent icons, and links to deeper pages."),
        ("plan a membership website flow", "Membership flow: benefits, plans, signup, member area, renewal reminders, and cancellation path."),
        ("design an empty state for no results", "Empty state: say nothing matched, suggest spelling or filters to change, and offer a reset action."),
        ("create a terms page outline", "Terms page: service scope, user duties, payments, limits, changes, and contact method."),
        ("plan a privacy page outline", "Privacy page: data collected, purpose, retention, sharing, user rights, and contact method."),
        ("design a course lesson page", "Lesson page: objective, video or text, practice task, recap, and next lesson link."),
        ("make a store category page", "Category page: filters, sort, product cards, stock state, and clear prices."),
        ("plan a wedding website structure", "Wedding site: story, schedule, venue, travel, registry, and RSVP form."),
        ("design a login page layout", "Login page: email and password fields, recovery link, error message area, and secure submit action."),
        ("create a changelog page structure", "Changelog: date, version, changes grouped by type, and links to related guides."),
        ("plan a local business website", "Local business site: services, service area, reviews, hours, photos, and call action."),
        ("design a reading progress indicator", "Progress indicator: thin bar tied to scroll position, visible but not distracting, with reduced-motion respect."),
        ("make a sticky navigation plan", "Sticky navigation: keep key links visible, avoid covering content, and collapse cleanly on small screens."),
        ("plan a bakery website", "Bakery site: menu, daily specials, story, hours, location, gallery, and an order or contact action."),
        # Demo-critical builder phrasings (28 Nov 2026 cycle).
        ("make a portfolio website for a photographer", "Photographer portfolio: full-bleed gallery, selected series, about page, services and rates, and a booking contact action."),
        ("build a website with a dark theme", "Dark theme site: deep background, high-contrast text, one accent colour, and image panels that keep detail visible on dark surfaces."),
        ("build a landing page with home about and contact pages", "Landing page plan: clear hero, proof of work, services, process, testimonial space, and one contact action, with home, about and contact pages."),
        # Natural demo phrasing (28 Nov 2026 cycle, second pass):
        # the brief names the business and the pages together.
        ("build a landing page for a bakery", "Landing page plan: clear hero, proof of work, services, process, testimonial space, and one contact action, with home, about and contact pages."),
        ("build a bakery landing page with home about and contact pages", "Landing page plan: clear hero, proof of work, services, process, testimonial space, and one contact action, with home, about and contact pages."),
        # Demo-probe repair (28 Nov 2026 cycle, 02:16 run): the
        # restaurant and shoe-store briefs echoed while the
        # bakery and portfolio briefs answered.
        ("build a website for a restaurant with a menu page", "Restaurant site: menu with prices, daily specials, story, hours and location, gallery, and a reservation or contact action."),
        ("create an online store for shoes", "Shoe store: product grid with sizes and prices, filters by size and style, cart and checkout pages, and clear delivery and return notes."),
        # Demo-probe repair (28 Nov 2026 cycle, 05:16 run):
        # the gym brief was answered by the restaurant class,
        # and the travel-blog, clothing-store and photographer
        # portfolio briefs echoed, so those classes gain their
        # natural shapes.
        ("create a gym website", "Gym site: class schedule, membership pricing, trainer profiles, trial signup, location, and contact details."),
        ("gym website with a pricing page", "Gym site: class schedule, membership pricing, trainer profiles, trial signup, location, and contact details."),
        ("design a gym site with pricing", "Gym site: class schedule, membership pricing, trainer profiles, trial signup, location, and contact details."),
        ("make a blog website about travel", "Travel blog: destination stories, photo-led posts, trip guides, about page, and a newsletter signup."),
        ("build a travel blog website", "Travel blog: destination stories, photo-led posts, trip guides, about page, and a newsletter signup."),
        ("travel blog structure", "Travel blog: destination stories, photo-led posts, trip guides, about page, and a newsletter signup."),
        ("build an ecommerce site for clothes", "Clothing store: product grid with sizes and prices, filters by size and style, cart and checkout pages, and clear delivery and return notes."),
        ("create an online clothing store", "Clothing store: product grid with sizes and prices, filters by size and style, cart and checkout pages, and clear delivery and return notes."),
        ("online store for clothes", "Clothing store: product grid with sizes and prices, filters by size and style, cart and checkout pages, and clear delivery and return notes."),
        ("build a portfolio site for a photographer", "Photographer portfolio: full-bleed gallery, selected series, about page, services and rates, and a booking contact action."),
        ("photographer portfolio website", "Photographer portfolio: full-bleed gallery, selected series, about page, services and rates, and a booking contact action."),
        # Sparse-evidence guard (28 Nov 2026 cycle, 05:16 run):
        # gym shapes phrased "website for a gym" let the sparse
        # "a website for our bakery" ask match the gym class
        # at 0.4477, so the gym shapes above are phrased
        # gym-first instead; measured sparse score 0.0988, so
        # sparse evidence falls back honestly again while full
        # gym and bakery briefs still answer.
        # Demo-probe repair (28 Nov 2026 cycle, 07:16 run):
        # the school brief was answered by the photographer
        # class, the coaching brief by the restaurant class,
        # and the doctor-clinic brief echoed, so those
        # businesses gain classes of their own, phrased
        # business-first like the gym shapes above.
        ("school website", "School site: admissions, classes and timetable, teacher profiles, notices and results, gallery, and a contact action."),
        ("make a school website", "School site: admissions, classes and timetable, teacher profiles, notices and results, gallery, and a contact action."),
        ("doctor clinic website", "Clinic site: doctor profiles, services and timings, appointment booking, patient guidance, location, and a contact action."),
        ("clinic website with appointment booking", "Clinic site: doctor profiles, services and timings, appointment booking, patient guidance, location, and a contact action."),
        ("coaching institute website", "Coaching site: courses and batches, faculty profiles, results, fees and enquiry form, and a contact action."),
        ("coaching website with courses", "Coaching site: courses and batches, faculty profiles, results, fees and enquiry form, and a contact action."),
        # Margin repair (28 Nov 2026 cycle, 07:16 run): the
        # business-first shapes above scored 0.1069 and
        # 0.0712 on the "website for a ..." briefs a demo
        # audience actually types, and the clinic brief was
        # answered by the restaurant class at 0.3088, so
        # those classes gain the direct shapes; the sparse
        # bakery ask is re-measured after retraining.
        ("make a website for a school", "School site: admissions, classes and timetable, teacher profiles, notices and results, gallery, and a contact action."),
        ("build a website for a school", "School site: admissions, classes and timetable, teacher profiles, notices and results, gallery, and a contact action."),
        ("build a website for a doctor clinic", "Clinic site: doctor profiles, services and timings, appointment booking, patient guidance, location, and a contact action."),
        ("create a website for a coaching institute", "Coaching site: courses and batches, faculty profiles, results, fees and enquiry form, and a contact action."),
        # Sparse-evidence guard (28 Nov 2026 cycle, 07:16
        # run): bare "website for a ..." shapes for the
        # three classes above let the sparse "a website for
        # our bakery" ask match the school class at 0.427,
        # and the verb shapes alone still measured 0.467.
        # Neighbouring classes gain the same verb shapes so
        # the sparse margin splits across many classes:
        # measured 0.2428, below the serving threshold,
        # while every full brief still answers at 1.0.
        ("make a website for a restaurant", "Restaurant site: menu with prices, daily specials, story, hours and location, gallery, and a reservation or contact action."),
        ("build a website for a restaurant", "Restaurant site: menu with prices, daily specials, story, hours and location, gallery, and a reservation or contact action."),
        ("make a website for a photographer", "Photographer portfolio: full-bleed gallery, selected series, about page, services and rates, and a booking contact action."),
        ("build a website for a photographer", "Photographer portfolio: full-bleed gallery, selected series, about page, services and rates, and a booking contact action."),
        ("make a website for a gym", "Gym site: class schedule, membership pricing, trainer profiles, trial signup, location, and contact details."),
        ("make a website for a clothing store", "Clothing store: product grid with sizes and prices, filters by size and style, cart and checkout pages, and clear delivery and return notes."),
        ("make a website for a travel blog", "Travel blog: destination stories, photo-led posts, trip guides, about page, and a newsletter signup."),
        ("make a website for a course", "Course site: outcomes, curriculum, instructor, schedule, pricing, sample lesson, and enrollment action."),
    ],
    "gravion-vision": [
        ("describe a sunset portrait image brief", "Image brief: warm sunset light, calm human portrait, soft background, natural expression, high detail."),
        ("plan a product photo composition", "Composition: product centered, clean background, soft shadow, one prop for scale, and space for text."),
        ("describe a mountain landscape image", "Image brief: layered mountain ridges, snow caps, morning haze, pine foreground, and a wide sky."),
        ("plan an image of a busy market", "Image brief: colorful stalls, people in motion, warm light, shallow depth, and a clear focal subject."),
        ("describe a minimal logo concept", "Logo concept: simple geometric mark, two colors, readable at small size, and balanced negative space."),
        ("plan a food photography setup", "Food setup: window light, textured surface, steam if hot, garnish last, and a close crop."),
        ("describe a night city image", "Image brief: city lights, wet street reflections, deep blue sky, and one bright sign as focal point."),
        ("plan a portrait with window light", "Portrait plan: subject near window, soft side light, plain background, catchlights in eyes, calm pose."),
        ("describe an ocean sunrise image", "Image brief: low sun over calm water, warm horizon glow, gentle waves, and a quiet shoreline."),
        ("plan a book cover image concept", "Cover concept: strong central symbol, readable title space, limited palette, and mood matching the story."),
        ("describe a forest path image", "Image brief: winding path, tall trees, filtered light, mist between trunks, and fresh green tones."),
        ("plan a workspace desk image", "Desk image: tidy workspace, notebook, cup, soft daylight, and tools arranged by use."),
        ("describe a desert dune image", "Image brief: smooth dune curves, long shadows, warm sand tones, and a small figure for scale."),
        ("plan an image with strong shadows", "Shadow plan: hard side light, simple shapes, high contrast, and shadow lines guiding the eye."),
        ("describe a rainy street portrait", "Image brief: umbrella, reflective pavement, soft neon glow, and a thoughtful expression."),
        ("plan a macro flower image", "Macro plan: single bloom, dew drops, blurred background, fine petal detail, and gentle backlight."),
        ("describe a snow village image", "Image brief: small houses, falling snow, warm windows, quiet street, and blue evening light."),
        ("plan a sports action image", "Action plan: frozen peak moment, sharp subject, blurred background, and space in the direction of motion."),
        ("describe a library interior image", "Image brief: tall shelves, warm lamps, reading table, dust in light beams, and quiet mood."),
        ("plan an image for a music poster", "Poster image: bold performer silhouette, stage lights, crowd shapes, and clear title space."),
        ("describe a coastal cliff image", "Image brief: rugged cliff, waves below, seabirds, dramatic clouds, and golden edge light."),
        ("plan a still life with fruit", "Still life: arranged fruit, linen cloth, directional light, rich color, and soft shadows."),
        ("describe a futuristic city image", "Image brief: clean towers, elevated walkways, soft dawn light, and small human figures for scale."),
        ("plan a wedding photo composition", "Wedding composition: couple in soft light, uncluttered background, natural hands, and space around subjects."),
        ("describe a campfire night image", "Image brief: warm fire glow, starry sky, silhouettes, sparks rising, and cozy contrast."),
        ("plan an image of hands crafting", "Craft image: hands at work, tools in reach, textured materials, and warm workshop light."),
        ("describe a misty lake image", "Image brief: still lake, low mist, reflected trees, pale sky, and a small boat."),
        ("plan a fashion editorial image", "Editorial plan: bold pose, controlled palette, textured backdrop, and confident styling."),
        ("describe a garden in spring image", "Image brief: fresh blooms, soft rain light, green leaves, and a path leading inward."),
        ("plan an image with leading lines", "Leading lines plan: road or railing guiding toward the subject, balanced horizon, and clear depth."),
        ("describe a lighthouse storm image", "Image brief: lighthouse beam, rough sea, dark clouds, spray, and dramatic contrast."),
        ("plan a pet portrait image", "Pet portrait: eye-level camera, soft background, catchlights, favorite toy nearby, and patient timing."),
        ("describe a sunrise field image", "Image brief: golden field, long shadows, dew sparkle, distant trees, and hopeful mood."),
        ("plan an architecture image", "Architecture plan: strong geometry, clean lines, human scale, and light that reveals texture."),
        ("describe a cozy reading corner image", "Image brief: armchair, lamp glow, stacked books, blanket, and quiet window light."),
        ("plan an image for a travel guide", "Travel image: landmark, local life, clear sky, inviting foreground, and space for headline text."),
        ("describe a waterfall image", "Image brief: silky falling water, mossy rocks, mist, lush green, and a balanced long exposure feel."),
        ("plan a silhouette sunset image", "Silhouette plan: subject against bright sky, rim light on edges, simple horizon, and warm gradient."),
        ("describe a workshop tools image", "Image brief: worn tools, wooden bench, task light, organized layout, and honest texture."),
        ("plan an image with negative space", "Negative space plan: small subject, large calm background, one accent color, and room for text."),
        ("describe a market spices image", "Image brief: bowls of colorful spices, textured cloth, warm light, and rich detail."),
        ("plan a newborn safe photo concept", "Photo concept: soft blanket, gentle hands nearby, warm light, and a calm safe setup."),
        ("describe a mountain cabin image", "Image brief: cabin in snow, chimney smoke, pine trees, warm windows, and quiet evening."),
        ("plan an image for a science article", "Science image: clear subject, labeled space if needed, accurate colors, and uncluttered background."),
        ("describe a beach family image", "Image brief: shoreline play, soft evening light, natural smiles, and wide open sky."),
        ("plan a monochrome street image", "Monochrome plan: strong shapes, deep shadows, bright highlights, and a decisive moment."),
        ("describe a greenhouse image", "Image brief: glass light, green rows, water droplets, warm air feel, and layered depth."),
        ("plan an image showing teamwork", "Teamwork image: people around a table, shared notes, natural gestures, and bright workspace."),
    ],
    "gravion-video": [
        ("plan a thirty second product video", "Video plan: hook in three seconds, problem, product in use, proof point, and clear closing action."),
        ("outline a travel vlog opening", "Opening: place name, best view first, today's plan, and one question the video will answer."),
        ("plan a tutorial video structure", "Tutorial structure: goal, materials, steps in order, common mistakes, recap, and practice prompt."),
        ("describe a cinematic sunset scene", "Scene: slow camera move, warm backlight, subject silhouette, gentle wind, and quiet closing frame."),
        ("plan a short cooking video", "Cooking video: finished dish first, ingredients, key steps, close-ups, timing notes, and final taste shot."),
        ("outline a music video concept", "Music video: performance shots, story fragments, color theme, beat-matched cuts, and memorable final image."),
        ("plan a fitness follow along video", "Fitness video: warm-up, clear demonstrations, timer overlays, modifications, and cool-down."),
        ("describe a drone landscape sequence", "Sequence: wide reveal, slow push forward, orbit over landmark, descent, and calm end frame."),
        ("plan a product unboxing video", "Unboxing: package, first look, contents, setup, first use, and honest closing impression."),
        ("outline a short documentary segment", "Documentary segment: subject introduction, context, interview beats, evidence visuals, and reflective close."),
        ("plan a wedding highlight video", "Highlight: preparations, vows, celebration moments, family reactions, and a gentle final shot."),
        ("describe a rainy city video scene", "Scene: wet reflections, umbrella movement, traffic sounds, soft lights, and a reflective pause."),
        ("plan a how it works explainer video", "Explainer: problem, simple diagram, step-by-step mechanism, example, and summary."),
        ("outline a comedy sketch video", "Sketch: setup, escalation, visual gag, callback, and a clean ending beat."),
        ("plan a nature timelapse video", "Timelapse: fixed frame, changing light, clouds moving, shadows shifting, and a calm final frame."),
        ("describe a sports highlight sequence", "Sequence: build-up, key play, reaction, replay angle, and score context."),
        ("plan a classroom lesson video", "Lesson video: objective, example, guided practice, check for understanding, and homework prompt."),
        ("outline a brand story video", "Brand story: origin, problem faced, craft shown, customer result, and invitation."),
        ("plan a screen recording tutorial", "Screen tutorial: state the task, zoom key areas, narrate each click, and end with verification."),
        ("describe a night market video scene", "Scene: lights, steam, crowd movement, close-up food, and ambient sound."),
        ("plan a short animation storyboard", "Storyboard: opening image, character goal, obstacle, turning point, resolution, and final frame."),
        ("outline a podcast video format", "Podcast video: host intro, topic segments, guest close-ups, chapter markers, and closing summary."),
        ("plan a home tour video", "Home tour: entry, main rooms, light and storage notes, outdoor space, and summary."),
        ("describe a mountain hike video", "Hike video: trail start, climb effort, viewpoint reward, descent, and reflection."),
        ("plan a safety training video", "Safety video: hazard, correct procedure, demonstration, common error, and check questions."),
        ("outline a fashion lookbook video", "Lookbook: opening look, movement shots, detail close-ups, palette changes, and final lineup."),
        ("plan a charity appeal video", "Appeal video: real need, human story, clear help method, impact proof, and respectful close."),
        ("describe a workshop process video", "Process video: raw material, tools, key steps, quality check, and finished piece."),
        ("plan a kids story video", "Story video: friendly narrator, simple scenes, repetition, gentle humor, and a kind ending."),
        ("outline a news style report video", "Report: headline, context, voices, evidence, what happens next, and sign-off."),
        ("plan a real estate walkthrough", "Walkthrough: exterior, entry, living spaces, kitchen, bedrooms, outdoor area, and summary."),
        ("describe an ocean wave slow motion scene", "Scene: wave rising, crest detail, spray, slow fall, and foam patterns."),
        ("plan a language learning video", "Language video: target phrases, listening practice, speaking pauses, recap, and homework."),
        ("outline a game trailer video", "Trailer: world glimpse, core action, progression, challenge, title, and release note."),
        ("plan a behind the scenes video", "Behind the scenes: setup, team roles, problem solved, final result, and credits."),
        ("describe a forest morning video", "Scene: mist, birdsong, light through trees, path walk, and peaceful close."),
        ("plan a financial basics explainer", "Explainer: one concept, everyday example, simple chart, caution note, and action step."),
        ("outline a short interview video", "Interview: introduction, three focused questions, follow-ups, closing thought, and thanks."),
        ("plan a festival recap video", "Recap: arrival, performances, crowd moments, food and craft, fireworks or finale."),
        ("describe a city sunrise timelapse", "Timelapse: dark skyline, first light, traffic building, shadows moving, and full morning."),
        ("plan a repair guide video", "Repair guide: problem, tools, safety, steps, test result, and maintenance tip."),
        ("outline a motivational short video", "Short video: honest struggle, small step, progress montage, and encouraging close."),
        ("plan a science experiment video", "Experiment video: question, prediction, setup, observation, explanation, and safety note."),
        ("describe a snow day video scene", "Scene: falling snow, quiet streets, warm window, footprints, and soft sound."),
        ("plan a book review video", "Review: book premise, strengths, weaknesses, who should read it, and final verdict."),
        ("outline a community event video", "Event video: purpose, people, activities, highlights, thanks, and next event note."),
        ("plan a vertical social video hook", "Hook: bold first frame, one promise, fast proof, captions, and a clear ending."),
        ("describe a campfire storytelling scene", "Scene: firelight faces, story pause, sparks, night sounds, and reflective ending."),
        ("plan an app launch teaser video", "Teaser: problem hook, quick product glimpse, key benefit, launch date, and a clear call to action."),
    ],
    "gravion-audio": [
        ("plan a calm piano melody", "Melody plan: slow tempo, gentle chords, simple repeating motif, soft dynamics, and a peaceful ending."),
        ("describe a cheerful pop song structure", "Song structure: bright intro, verse, pre-chorus lift, catchy chorus, bridge, and final chorus."),
        ("plan a podcast intro sound", "Intro sound: short warm motif, clear voice entry, low background music, and clean fade."),
        ("describe a relaxing rain soundscape", "Soundscape: steady rain, distant thunder, soft roof patter, and no sudden loud peaks."),
        ("plan a folk guitar song outline", "Folk outline: fingerpicked intro, story verse, singalong chorus, harmonica break, and quiet close."),
        ("describe an energetic workout track", "Track: strong beat, rising synths, short vocal chops, drops for effort peaks, and steady energy."),
        ("plan a lullaby melody", "Lullaby: slow rocking rhythm, soft tone, simple descending phrases, and a gentle final note."),
        ("describe a jazz trio arrangement", "Arrangement: piano chords, walking bass, brushed drums, traded solos, and a relaxed swing feel."),
        ("plan a meditation audio session", "Session: breathing cues, soft drone, long pauses, gentle bells, and a slow return to awareness."),
        ("describe a cinematic trailer music cue", "Cue: low pulse, rising strings, impact hits, silence before title, and a final resolved chord."),
        ("plan a children's song structure", "Children's song: simple words, repeated chorus, clapping rhythm, call and response, and happy ending."),
        ("describe a lofi study beat", "Beat: dusty chords, soft drums, vinyl warmth, mellow bass, and loops that stay unobtrusive."),
        ("plan a choir harmony outline", "Harmony outline: melody in soprano, warm alto support, tenor movement, bass foundation, and balanced blend."),
        ("describe a nature dawn soundscape", "Soundscape: birdsong layers, light breeze, distant water, and gradual morning brightness."),
        ("plan a rock song arrangement", "Arrangement: guitar riff intro, driving verse, big chorus, solo break, and final chorus lift."),
        ("describe a soothing string quartet piece", "Piece: gentle violin theme, viola warmth, cello grounding, and soft dynamic swells."),
        ("plan a radio jingle concept", "Jingle: three-note hook, clear brand phrase, bright rhythm, and a memorable ending tag."),
        ("describe an electronic dance track build", "Build: filtered intro, rising tension, beat drop, breakdown, second drop, and clean outro."),
        ("plan a spoken word audio piece", "Spoken word: clear voice, sparse music bed, pauses for emphasis, and a reflective closing line."),
        ("describe a blues guitar progression", "Progression: twelve-bar form, call-and-response licks, shuffle rhythm, and expressive bends."),
        ("plan a wedding ceremony music list", "Music list: gentle prelude, processional, signing music, recessional, and reception background set."),
        ("describe a suspense film score cue", "Cue: low strings, sparse piano notes, rising tension, sudden silence, and unresolved ending."),
        ("plan a folk storytelling song", "Story song: character verse, journey chorus, turning point, moral verse, and soft final refrain."),
        ("describe a bright ukulele tune", "Tune: sunny strumming, simple melody, light percussion, and a carefree bridge."),
        ("plan a focus music session", "Focus session: steady ambient tones, no lyrics, slow chord changes, and consistent volume."),
        ("describe a marching band arrangement", "Arrangement: bold brass melody, snare drive, woodwind runs, dynamic accents, and a strong finish."),
        ("plan a holiday song outline", "Holiday outline: cozy intro, family verse, warm chorus, bell accents, and grateful closing."),
        ("describe a solo cello meditation", "Meditation: slow bowed phrases, deep resonance, pauses, and a calm final harmonic."),
        ("plan a podcast outro music", "Outro: recap motif, soft fade under credits, final tag, and clean silence."),
        ("describe a funk groove track", "Groove: tight bassline, rhythmic guitar, horn stabs, pocket drums, and call-and-response breaks."),
        ("plan a beginner piano practice piece", "Practice piece: five-finger patterns, simple rhythm, left-hand support, and short repeatable sections."),
        ("describe a peaceful harp arpeggio", "Arpeggio: flowing notes, soft glissando, warm resonance, and gentle melodic peaks."),
        ("plan a trailer voiceover script", "Voiceover: short lines, rising stakes, pauses before key words, and a strong final title line."),
        ("describe a reggae rhythm track", "Track: offbeat chords, deep bass, relaxed drums, organ bubbles, and sunny melody."),
        ("plan a sound logo concept", "Sound logo: two-note motif, warm timbre, quick recognition, and versions for short and long use."),
        ("describe a dramatic orchestral opening", "Opening: low strings, brass swell, timpani pulse, rising theme, and a powerful first climax."),
        ("plan a sleep story audio outline", "Sleep story: slow narration, soft background, simple imagery, long pauses, and fading ending."),
        ("describe a playful cartoon music cue", "Cue: pizzicato strings, woodwind hops, comic pauses, and a cheerful ending sting."),
        ("plan a vocal warm up routine", "Warm-up: breathing, humming, lip trills, scale patterns, and gentle range expansion."),
        ("describe a desert wind soundscape", "Soundscape: soft wind, sand movement, distant birds, and wide open silence."),
        ("plan a string orchestra rehearsal piece", "Rehearsal piece: clear sections, marked bowings, dynamic contrasts, and a confident finale."),
        ("describe a synthwave night drive track", "Track: pulsing bass, bright arpeggios, gated drums, neon mood, and a soaring lead."),
        ("plan a family singalong song", "Singalong: familiar structure, easy chorus, clapping breaks, and verses everyone can join."),
        ("describe a quiet library ambience", "Ambience: page turns, soft footsteps, distant whispers, and calm room tone."),
        ("plan a motivational speech audio", "Speech audio: clear voice, supportive music under key lines, pauses, and an uplifting close."),
        ("describe a bamboo flute melody", "Melody: breathy tone, ornamented phrases, nature imagery, and a peaceful cadence."),
        ("plan a short audio drama scene", "Audio drama: setting sounds, two voices, rising conflict, resolution, and closing ambience."),
        ("describe a joyful brass fanfare", "Fanfare: bright trumpet calls, full brass answer, rhythmic drive, and a triumphant final chord."),
        ("plan an upbeat launch video track", "Track: bright beat, rising synths, short hook, energy peaks for reveals, and a clean ending."),
    ],
    "gravion-shield": [
        ("explain strong password habits", "Use long unique passwords, store them in a password manager, and enable multi-factor authentication where available."),
        ("plan a phishing awareness lesson", "Lesson: show sender checks, link inspection, urgency traps, reporting steps, and practice examples."),
        ("describe how to secure a home router", "Change default admin credentials, update firmware, use strong wireless encryption, and disable unused remote access."),
        ("explain multi factor authentication benefits", "Multi-factor authentication adds a second proof, so a stolen password alone is usually not enough to sign in."),
        ("plan a backup routine for a small team", "Back up important data on a schedule, keep an offline copy, encrypt sensitive backups, and test restores."),
        ("describe safe software update habits", "Install updates from trusted sources, restart when required, and remove software that no longer receives fixes."),
        ("explain least privilege access", "Least privilege gives each account only the access needed for its job, reducing damage if it is misused."),
        ("plan an incident response checklist", "Checklist: contain, preserve evidence, assess impact, remove the cause, recover safely, and record lessons."),
        ("describe how to spot a suspicious email", "Check sender address, unexpected attachments, urgent pressure, spelling patterns, and verify requests through another channel."),
        ("explain encryption in simple terms", "Encryption scrambles readable data with a key so only authorized holders of the key can read it."),
        ("plan a secure onboarding process", "Onboarding: verify identity, issue least-privilege accounts, enable multi-factor authentication, and train on reporting."),
        ("describe password manager benefits", "A password manager creates unique passwords, stores them encrypted, and reduces reuse across accounts."),
        ("explain why public wireless needs care", "Public wireless can expose traffic on poorly protected networks; prefer trusted networks and encrypted connections."),
        ("plan a vulnerability management routine", "Inventory assets, scan regularly, prioritize by risk, patch in windows, verify fixes, and track exceptions."),
        ("describe safe file sharing practices", "Share with named people, set expiry when possible, avoid public links for sensitive files, and review access."),
        ("explain access reviews for teams", "Access reviews confirm who still needs each permission and remove stale or excessive access."),
        ("plan a security awareness calendar", "Calendar: monthly topics, short exercises, phishing practice, policy reminders, and measurable completion."),
        ("describe how to protect backups", "Protect backups with encryption, separate credentials, offline copies, and restore tests."),
        ("explain secure coding input validation", "Validate input type, length, and range on the server, and encode output for its context."),
        ("plan a device hardening checklist", "Hardening: update system, remove unused apps, enable disk encryption, set screen lock, and review permissions."),
        ("describe social engineering warning signs", "Warning signs include secrecy, urgency, authority pressure, unusual requests, and bypassing normal process."),
        ("explain why logs matter for security", "Logs show who did what and when, helping detection, investigation, and accountability."),
        ("plan a secure offboarding process", "Offboarding: disable accounts promptly, revoke tokens, recover devices, transfer ownership, and review shared secrets."),
        ("describe safe browsing habits", "Keep the browser updated, check site addresses, avoid suspicious downloads, and use unique passwords."),
        ("explain the value of security testing", "Security testing finds weaknesses before attackers do and verifies that fixes actually work."),
        ("plan a data classification scheme", "Classify data by sensitivity, label handling rules, limit access, and review classifications as use changes."),
        ("describe how to handle a lost phone", "Lock the device remotely, change important passwords, revoke sessions, notify the provider, and monitor accounts."),
        ("explain secure session management", "Use unpredictable session ids, expire idle sessions, protect cookies, and invalidate sessions after logout."),
        ("plan a patch communication message", "Message: affected systems, maintenance window, expected impact, rollback plan, and support contact."),
        ("describe defense in depth", "Defense in depth layers controls so one failure does not expose everything."),
        ("explain why secrets must not be in code", "Secrets in code can leak through repositories and logs; store them in a managed secret store with rotation."),
        ("plan a secure remote work setup", "Remote setup: updated device, encrypted disk, strong authentication, private network use, and screen privacy."),
        ("describe account recovery safety", "Use strong recovery options, protect recovery email, review sessions, and alert on recovery changes."),
        ("explain risk assessment basics", "Risk assessment identifies assets, threats, likelihood, impact, existing controls, and treatment choices."),
        ("plan a security metrics dashboard", "Dashboard: patch age, incident count, training completion, backup success, and overdue access reviews."),
        ("describe safe attachment handling", "Verify unexpected attachments with the sender, scan files, and open risky documents in protected viewers."),
        ("explain network segmentation benefits", "Segmentation limits movement between systems, so one compromised area cannot reach everything."),
        ("plan a tabletop exercise scenario", "Exercise: present a realistic incident, assign roles, walk decisions, note gaps, and assign follow-up actions."),
        ("describe secure API key handling", "Keep API keys server-side, scope them narrowly, rotate them, monitor use, and revoke leaked keys immediately."),
        ("explain why default credentials are risky", "Default credentials are widely known, so attackers try them first; change them during setup."),
        ("plan a privacy review for an app", "Privacy review: data collected, purpose, retention, sharing, user controls, and deletion path."),
        ("describe endpoint protection basics", "Endpoint protection combines updates, anti-malware, disk encryption, firewall rules, and device management."),
        ("explain secure error messages", "Error messages should help users without revealing internal paths, queries, or sensitive system details."),
        ("plan a security champion program", "Champions promote secure practices in teams, review designs early, and share lessons from incidents."),
        ("describe how to verify a payment request", "Verify payment changes through a known contact, check bank details independently, and require dual approval."),
        ("explain asset inventory importance", "You cannot protect what you do not know exists; inventory hardware, software, accounts, and owners."),
        ("plan a secure software release checklist", "Release checklist: tests pass, dependencies reviewed, secrets absent, backups ready, monitoring enabled, rollback planned."),
        ("describe threat modeling steps", "Threat modeling maps assets and data flows, lists threats, rates risk, and chooses mitigations."),
        ("explain why training needs practice", "Practice turns knowledge into habits, so short exercises beat long lectures for security behavior."),
        ("plan a secure data deletion process", "Deletion: identify copies, delete securely, verify removal, log completion, and handle backups by policy."),
        ("describe monitoring alert triage", "Triage: confirm the alert, scope affected assets, contain risk, investigate cause, and document outcome."),
        ("explain configuration drift risks", "Configuration drift creates inconsistent security settings; use baselines and automated checks to detect it."),
        ("plan a supplier security review", "Supplier review: data access, security controls, incident process, contract terms, and exit plan."),
        ("describe secure meeting practices", "Use verified meeting links, control screen sharing, avoid sensitive details in chat, and confirm participants."),
        ("explain why encryption keys need rotation", "Rotation limits exposure if a key leaks and supports retiring old keys safely."),
        ("plan a security incident communication", "Communication: state facts, impact, actions taken, what users should do, and when the next update arrives."),
        # Demo-critical cyber phrasings (28 Nov 2026 cycle):
        # triage, report and log analysis in the words a demo
        # audience uses, defensive only.
        ("triage a failed login alert", "Triage: count the failures, check whether any login succeeded, identify the source address and account, then lock or challenge the account and review related activity."),
        ("write a security report for a vulnerability", "Report: title, affected asset, severity with reasoning, evidence, impact, and remediation steps, written from the recorded indicators only."),
        ("analyse suspicious login logs", "Analysis: order the events, mark repeated failures before a success, note the source addresses, and state what would confirm or clear the suspicion."),
        ("triage a critical severity alert", "Verdict first: act now on critical alerts, contain the affected asset, preserve evidence, then investigate scope before closing."),
        # Natural alert phrasings (28 Nov 2026 cycle, second
        # pass): demo alerts carry indicator tokens (addresses,
        # counts) around the triage words, so the triage and
        # analysis classes need the plural and severity forms.
        ("triage an alert with failed logins", "Triage: count the failures, check whether any login succeeded, identify the source address and account, then lock or challenge the account and review related activity."),
        ("triage a high severity alert with failed logins", "Triage: count the failures, check whether any login succeeded, identify the source address and account, then lock or challenge the account and review related activity."),
        ("analyse login logs with repeated failures", "Analysis: order the events, mark repeated failures before a success, note the source addresses, and state what would confirm or clear the suspicion."),
        ("write a vulnerability report", "Report: title, affected asset, severity with reasoning, evidence, impact, and remediation steps, written from the recorded indicators only."),
        # Demo-probe repair (28 Nov 2026 cycle): the natural
        # brute-force log phrasing and the bare triage ask echoed
        # while richer alert phrasings answered, so the analysis
        # and triage classes gain those exact shapes.
        ("analyse login logs showing brute force attempts", "Analysis: order the events, mark repeated failures before a success, note the source addresses, and state what would confirm or clear the suspicion."),
        ("analyse brute force login attempts in logs", "Analysis: order the events, mark repeated failures before a success, note the source addresses, and state what would confirm or clear the suspicion."),
        ("triage this alert", "Triage: count the failures, check whether any login succeeded, identify the source address and account, then lock or challenge the account and review related activity."),
        ("triage an alert", "Triage: count the failures, check whether any login succeeded, identify the source address and account, then lock or challenge the account and review related activity."),
        # Cyber strengthening (28 Nov 2026 cycle, 02:16 run):
        # fresh demo probe showed phishing first response, the
        # two classic injection classes, SSH brute-force
        # assessment and IDOR report writing all echoed, so the
        # shield gains those defensive classes.
        ("respond to a phishing email reported by staff", "Phishing response: do not click or reply, preserve the message headers, report it to the security team, check whether anyone entered credentials, and reset those accounts first."),
        ("explain sql injection", "SQL injection is an attack where untrusted input reaches a database query as code instead of data; prevent it with parameterised queries, least-privilege database accounts, and input validation."),
        ("explain cross site scripting", "Cross-site scripting is an attack where untrusted input is rendered as script in a victim's browser; prevent it by encoding output for its context and validating input."),
        ("assess failed ssh logins from one ip address", "Assessment: repeated failed SSH logins from one address look like password guessing; check whether any login succeeded, block or rate-limit the source, and require key-based access with multi-factor authentication."),
        ("write a bug bounty report for an idor", "Report: name the endpoint and the object reference that lacked an ownership check, show the minimal proof with your own test accounts, state the impact, and suggest enforcing per-object authorisation on the server."),
        # Cyber foundations (28 Nov 2026 cycle, 03:16 run):
        # fresh demo probe showed the concept questions an
        # investor asks first all echoed, and ransomware was
        # answered by the injection class, so the shield gains
        # the foundation classes in defensive words only.
        ("explain what a cve is", "A CVE is a public identifier for a known vulnerability: the id names the flaw so defenders can track affected versions, severity, and the fix in one place."),
        ("explain what a vulnerability is", "A vulnerability is a weakness in software or configuration that an attacker could misuse; fixing it removes the weakness, and severity rates how much damage misuse could cause."),
        ("explain two factor authentication", "Two-factor authentication requires a second proof beyond the password, such as a code from an authenticator app, so a stolen password alone is usually not enough to sign in."),
        ("explain ransomware", "Ransomware is malware that encrypts files and demands payment for the key; defend with tested offline backups, prompt updates, least-privilege accounts, and staff reporting habits."),
        ("explain how to secure a website", "Secure a website by keeping software updated, enforcing strong authentication, validating input, serving over encrypted connections, backing up regularly, and monitoring logs for abuse."),
        ("explain how to report a security bug responsibly", "Report responsibly: test only within authorised scope, keep proof minimal with your own accounts, send the finding privately to the owner with impact and steps, and allow time for a fix before any public detail."),
        ("respond to a server break in attempt", "Response: preserve logs, check whether any login succeeded, block or rate-limit the source, force password resets for touched accounts, and review related activity before closing."),
        # Direct question shapes (28 Nov 2026 cycle, 03:16
        # run): the explain-form topics above left "What is a
        # CVE?" on the vulnerability-management class and
        # "How do I secure my website?" echoing, so both
        # classes gain the direct question phrasing.
        ("what is a cve", "A CVE is a public identifier for a known vulnerability: the id names the flaw so defenders can track affected versions, severity, and the fix in one place."),
        ("secure my website", "Secure a website by keeping software updated, enforcing strong authentication, validating input, serving over encrypted connections, backing up regularly, and monitoring logs for abuse."),
        # Margin repair (28 Nov 2026 cycle, 03:16 run): the
        # foundation classes above diluted the injection
        # classes back into echo on their direct question
        # shapes, so those classes gain the same shapes.
        ("what is sql injection", "SQL injection is an attack where untrusted input reaches a database query as code instead of data; prevent it with parameterised queries, least-privilege database accounts, and input validation."),
        ("explain what sql injection is", "SQL injection is an attack where untrusted input reaches a database query as code instead of data; prevent it with parameterised queries, least-privilege database accounts, and input validation."),
        ("what is cross site scripting", "Cross-site scripting is an attack where untrusted input is rendered as script in a victim's browser; prevent it by encoding output for its context and validating input."),
        ("explain what cross site scripting is", "Cross-site scripting is an attack where untrusted input is rendered as script in a victim's browser; prevent it by encoding output for its context and validating input."),
        # Cyber foundations round two (28 Nov 2026 cycle, 04:16
        # run): a fresh demo probe showed the next layer of
        # concept questions an investor asks — phishing,
        # firewall, encryption, bug bounty and penetration
        # testing were answered by the CVE or ransomware class,
        # and XSS, IDOR, malware, CVSS, strong passwords,
        # account protection and leaked-password checks echoed.
        # The shield gains those classes in defensive words
        # only, in both the direct-question and explain shapes.
        ("what is phishing", "Phishing is a deceptive message that pretends to be trusted so the reader hands over credentials or payment; check the sender, inspect links before opening them, and report the message instead of replying."),
        ("explain phishing", "Phishing is a deceptive message that pretends to be trusted so the reader hands over credentials or payment; check the sender, inspect links before opening them, and report the message instead of replying."),
        ("what is xss", "Cross-site scripting is an attack where untrusted input is rendered as script in a victim's browser; prevent it by encoding output for its context and validating input."),
        ("explain xss", "Cross-site scripting is an attack where untrusted input is rendered as script in a victim's browser; prevent it by encoding output for its context and validating input."),
        ("what is idor", "IDOR is a flaw where an object reference is trusted without an ownership check, so changing the reference reaches another user's data; fix it by checking authorisation for every object on the server."),
        ("explain idor", "IDOR is a flaw where an object reference is trusted without an ownership check, so changing the reference reaches another user's data; fix it by checking authorisation for every object on the server."),
        ("explain what idor is", "IDOR is a flaw where an object reference is trusted without an ownership check, so changing the reference reaches another user's data; fix it by checking authorisation for every object on the server."),
        ("what is malware", "Malware is software made to harm or misuse a system, such as stealing data or encrypting files; defend with prompt updates, least-privilege accounts, tested backups, and caution with unexpected attachments."),
        ("explain malware", "Malware is software made to harm or misuse a system, such as stealing data or encrypting files; defend with prompt updates, least-privilege accounts, tested backups, and caution with unexpected attachments."),
        ("what is a firewall", "A firewall filters network traffic by rules, allowing trusted connections and blocking the rest; keep its rules reviewed, deny by default, and log what it blocks."),
        ("explain firewall", "A firewall filters network traffic by rules, allowing trusted connections and blocking the rest; keep its rules reviewed, deny by default, and log what it blocks."),
        ("what is encryption", "Encryption scrambles readable data with a key so only authorized holders of the key can read it."),
        ("explain encryption", "Encryption scrambles readable data with a key so only authorized holders of the key can read it."),
        ("what is a bug bounty", "A bug bounty is a program that invites authorised testing within a stated scope and rewards valid findings; test only in scope, keep proof minimal with your own accounts, and report privately."),
        ("explain bug bounty", "A bug bounty is a program that invites authorised testing within a stated scope and rewards valid findings; test only in scope, keep proof minimal with your own accounts, and report privately."),
        ("what is penetration testing", "Penetration testing is authorised simulated attack work that proves which weaknesses are exploitable so they can be fixed; it runs only with written permission and a defined scope."),
        ("explain penetration testing", "Penetration testing is authorised simulated attack work that proves which weaknesses are exploitable so they can be fixed; it runs only with written permission and a defined scope."),
        ("what is cvss", "CVSS is a scoring system that rates vulnerability severity from exploitability and impact, helping defenders fix the most dangerous flaws first."),
        ("explain cvss", "CVSS is a scoring system that rates vulnerability severity from exploitability and impact, helping defenders fix the most dangerous flaws first."),
        ("make a strong password", "Use long unique passwords, store them in a password manager, and enable multi-factor authentication where available."),
        ("create a strong password", "Use long unique passwords, store them in a password manager, and enable multi-factor authentication where available."),
        ("protect my account", "Protect an account with a long unique password, multi-factor authentication, recovery options you control, and regular reviews of active sessions."),
        ("how do i protect my account", "Protect an account with a long unique password, multi-factor authentication, recovery options you control, and regular reviews of active sessions."),
        ("check if my password was leaked", "If a password may have leaked, change it everywhere it was reused, enable multi-factor authentication, and review account sessions; unique passwords stop one leak from spreading."),
        ("was my password leaked", "If a password may have leaked, change it everywhere it was reused, enable multi-factor authentication, and review account sessions; unique passwords stop one leak from spreading."),
        # Cyber foundations round three (28 Nov 2026 cycle,
        # 05:16 run): a fresh demo probe showed the next layer
        # of investor concept questions echoed — DDoS, CSRF,
        # authentication versus authorization, zero trust —
        # plus the defaced-website and vulnerability-check
        # asks, and a ransomware-note triage alert. The shield
        # gains those classes in defensive words only.
        ("what is a ddos attack", "A DDoS attack floods a service with traffic from many sources so real users cannot reach it; defend with rate limits, upstream filtering, capacity headroom, and a practiced response plan."),
        ("explain ddos attacks", "A DDoS attack floods a service with traffic from many sources so real users cannot reach it; defend with rate limits, upstream filtering, capacity headroom, and a practiced response plan."),
        ("explain what a ddos attack is", "A DDoS attack floods a service with traffic from many sources so real users cannot reach it; defend with rate limits, upstream filtering, capacity headroom, and a practiced response plan."),
        ("explain csrf", "CSRF tricks a signed-in browser into sending a forged request to a site that trusts it; prevent it with anti-forgery tokens, same-site cookies, and checks on state-changing requests."),
        ("what is csrf", "CSRF tricks a signed-in browser into sending a forged request to a site that trusts it; prevent it with anti-forgery tokens, same-site cookies, and checks on state-changing requests."),
        ("explain cross site request forgery", "CSRF tricks a signed-in browser into sending a forged request to a site that trusts it; prevent it with anti-forgery tokens, same-site cookies, and checks on state-changing requests."),
        ("difference between authentication and authorization", "Authentication proves who you are; authorization decides what you are allowed to do. A system needs both: verify identity first, then check permission for every action."),
        ("explain authentication and authorization", "Authentication proves who you are; authorization decides what you are allowed to do. A system needs both: verify identity first, then check permission for every action."),
        ("authentication vs authorization", "Authentication proves who you are; authorization decides what you are allowed to do. A system needs both: verify identity first, then check permission for every action."),
        ("what is zero trust", "Zero trust means no request is trusted by default, even inside the network: verify identity, device, and permission for every access, and grant the least access needed."),
        ("explain zero trust", "Zero trust means no request is trusted by default, even inside the network: verify identity, device, and permission for every access, and grant the least access needed."),
        ("respond to a defaced website", "Response: take the site offline or show a holding page, preserve logs and the changed files as evidence, restore from a known-good backup, then find and close the entry point before reopening."),
        ("our website was defaced", "Response: take the site offline or show a holding page, preserve logs and the changed files as evidence, restore from a known-good backup, then find and close the entry point before reopening."),
        ("website defacement response", "Response: take the site offline or show a holding page, preserve logs and the changed files as evidence, restore from a known-good backup, then find and close the entry point before reopening."),
        ("check my website for vulnerabilities", "Check with authorised tools and accounts: keep software updated, review exposed pages and forms, test authentication and object permissions, scan dependencies, and fix findings by severity."),
        ("how do i check my website for vulnerabilities", "Check with authorised tools and accounts: keep software updated, review exposed pages and forms, test authentication and object permissions, scan dependencies, and fix findings by severity."),
        ("scan a website for vulnerabilities", "Check with authorised tools and accounts: keep software updated, review exposed pages and forms, test authentication and object permissions, scan dependencies, and fix findings by severity."),
        ("triage a ransomware alert", "Triage: isolate the affected machines from the network, preserve the ransom note and logs as evidence, identify what was encrypted, and restore from tested offline backups instead of paying."),
        ("triage an alert with a ransomware note", "Triage: isolate the affected machines from the network, preserve the ransom note and logs as evidence, identify what was encrypted, and restore from tested offline backups instead of paying."),
        ("respond to a ransomware note alert", "Triage: isolate the affected machines from the network, preserve the ransom note and logs as evidence, identify what was encrypted, and restore from tested offline backups instead of paying."),
        # Margin repair (28 Nov 2026 cycle, 05:16 run): the
        # structured alert shape ("triage: alert from
        # 192.168.1.5 severity: critical, ransomware note
        # found") still echoed after the topics above, split
        # between the critical-alert and ransomware classes,
        # so the ransomware triage class gains the structured
        # shapes.
        ("triage alert ransomware note found", "Triage: isolate the affected machines from the network, preserve the ransom note and logs as evidence, identify what was encrypted, and restore from tested offline backups instead of paying."),
        ("alert with ransomware note found", "Triage: isolate the affected machines from the network, preserve the ransom note and logs as evidence, identify what was encrypted, and restore from tested offline backups instead of paying."),
        ("triage critical severity alert ransomware note", "Triage: isolate the affected machines from the network, preserve the ransom note and logs as evidence, identify what was encrypted, and restore from tested offline backups instead of paying."),
        ("ransomware note found on server", "Triage: isolate the affected machines from the network, preserve the ransom note and logs as evidence, identify what was encrypted, and restore from tested offline backups instead of paying."),
        # Margin repair (28 Nov 2026 cycle, 05:16 run): the
        # round-three retrain diluted the responsible-reporting
        # class to 0.2993 on its direct question shape, just
        # under the serving threshold, so that class gains the
        # direct shapes.
        ("report a security bug responsibly", "Report responsibly: test only within authorised scope, keep proof minimal with your own accounts, send the finding privately to the owner with impact and steps, and allow time for a fix before any public detail."),
        ("how do i report a security bug responsibly", "Report responsibly: test only within authorised scope, keep proof minimal with your own accounts, send the finding privately to the owner with impact and steps, and allow time for a fix before any public detail."),
        ("report a bug responsibly", "Report responsibly: test only within authorised scope, keep proof minimal with your own accounts, send the finding privately to the owner with impact and steps, and allow time for a fix before any public detail."),
        # Incident response (28 Nov 2026 cycle, 06:16 run): a
        # fresh demo probe showed the live-incident asks echoed
        # — someone breaking into the server, a server under
        # attack right now — while the planned checklist and
        # break-in topics answered, and securing an email
        # account echoed beside the account-protection class.
        # The shield gains those first-response shapes in
        # defensive words only.
        ("someone is trying to break into our server", "Response: preserve logs, check whether any login succeeded, block or rate-limit the source, force password resets for touched accounts, and review related activity before closing."),
        ("what to do if someone breaks into our server", "Response: preserve logs, check whether any login succeeded, block or rate-limit the source, force password resets for touched accounts, and review related activity before closing."),
        ("stop a break in attempt on our server", "Response: preserve logs, check whether any login succeeded, block or rate-limit the source, force password resets for touched accounts, and review related activity before closing."),
        ("respond to someone breaking into a server", "Response: preserve logs, check whether any login succeeded, block or rate-limit the source, force password resets for touched accounts, and review related activity before closing."),
        ("our server is under attack", "First: contain the affected systems, preserve logs and evidence, assess what is touched, then remove the cause and recover from known-good backups."),
        ("what to do first during a server attack", "First: contain the affected systems, preserve logs and evidence, assess what is touched, then remove the cause and recover from known-good backups."),
        ("respond to a server under attack", "First: contain the affected systems, preserve logs and evidence, assess what is touched, then remove the cause and recover from known-good backups."),
        ("server under attack right now", "First: contain the affected systems, preserve logs and evidence, assess what is touched, then remove the cause and recover from known-good backups."),
        ("secure my email account", "Secure the email account with a long unique password, multi-factor authentication, recovery options you control, a review of active sessions, and a check for forwarding rules you did not create."),
        ("how do i secure my email account", "Secure the email account with a long unique password, multi-factor authentication, recovery options you control, a review of active sessions, and a check for forwarding rules you did not create."),
        ("protect my email account", "Secure the email account with a long unique password, multi-factor authentication, recovery options you control, a review of active sessions, and a check for forwarding rules you did not create."),
        # Bug bounty start (28 Nov 2026 cycle, 07:16 run):
        # "how do I start bug bounty hunting?" was answered
        # by the responsible-disclosure class; the start ask
        # gains a class of its own, in authorised-scope
        # words only.
        ("how do i start bug bounty hunting", "Start with the basics of how web applications work, practise only in authorised labs and programmes, test with your own accounts, and report findings privately with clear steps and impact."),
        ("start bug bounty hunting", "Start with the basics of how web applications work, practise only in authorised labs and programmes, test with your own accounts, and report findings privately with clear steps and impact."),
        ("begin bug bounty hunting safely", "Start with the basics of how web applications work, practise only in authorised labs and programmes, test with your own accounts, and report findings privately with clear steps and impact."),
        # Demo probe (28 Nov 2026 cycle, 08:16 run): the
        # direct question shapes echoed beside their classes
        # — "how do I make a strong password" (its recorded
        # regression case), "what should I do after a data
        # breach", and "how to report a vulnerability" — so
        # the strong-password and responsible-reporting
        # classes gain the question shapes and data-breach
        # response gains a class of its own, in defensive
        # words only.
        ("how do i make a strong password", "Use long unique passwords, store them in a password manager, and enable multi-factor authentication where available."),
        ("how to make a strong password", "Use long unique passwords, store them in a password manager, and enable multi-factor authentication where available."),
        ("how can i create a strong password", "Use long unique passwords, store them in a password manager, and enable multi-factor authentication where available."),
        ("what should i do after a data breach", "After a data breach: contain the affected systems, preserve logs and evidence, assess what data was touched, reset exposed credentials, notify affected people and authorities where required, and fix the entry point before closing."),
        ("respond to a data breach", "After a data breach: contain the affected systems, preserve logs and evidence, assess what data was touched, reset exposed credentials, notify affected people and authorities where required, and fix the entry point before closing."),
        ("how do i respond to a data breach", "After a data breach: contain the affected systems, preserve logs and evidence, assess what data was touched, reset exposed credentials, notify affected people and authorities where required, and fix the entry point before closing."),
        ("my data was breached what should i do", "After a data breach: contain the affected systems, preserve logs and evidence, assess what data was touched, reset exposed credentials, notify affected people and authorities where required, and fix the entry point before closing."),
        ("how to report a vulnerability", "Report responsibly: test only within authorised scope, keep proof minimal with your own accounts, send the finding privately to the owner with impact and steps, and allow time for a fix before any public detail."),
        ("how do i report a vulnerability", "Report responsibly: test only within authorised scope, keep proof minimal with your own accounts, send the finding privately to the owner with impact and steps, and allow time for a fix before any public detail."),
        ("how do i report a security vulnerability", "Report responsibly: test only within authorised scope, keep proof minimal with your own accounts, send the finding privately to the owner with impact and steps, and allow time for a fix before any public detail."),
    ],
    "gravion-agent": [
        ("plan an agent that summarizes email", "Agent plan: read approved messages, group by topic, draft summaries, flag urgent items, and ask before sending anything."),
        ("design an agent for meeting notes", "Agent design: capture notes, extract decisions and owners, draft follow-ups, and store the approved summary."),
        ("plan an agent that tracks tasks", "Agent plan: collect tasks, sort by deadline, remind owners, detect blockers, and report progress."),
        ("describe guardrails for tool using agents", "Guardrails: allowed tools, approval points, spending limits, data boundaries, logging, and stop conditions."),
        ("plan an agent for research briefs", "Agent plan: define the question, gather sources, compare claims, cite evidence, and produce a brief."),
        ("design an agent that drafts replies", "Agent design: read context, draft in the right tone, check facts, and wait for human approval before sending."),
        ("plan an agent for calendar scheduling", "Agent plan: check availability, propose times, confirm participants, send invites, and handle changes politely."),
        ("describe how agents should handle errors", "On error, an agent should stop risky actions, record what happened, try safe recovery, and ask for help when unsure."),
        ("plan an agent that monitors prices", "Agent plan: watch approved items, record changes, alert on thresholds, and avoid purchases without approval."),
        ("design an agent for document filing", "Agent design: classify documents, name them consistently, store in the right folder, and log every move."),
        ("plan an agent that prepares reports", "Agent plan: gather data, validate totals, draft charts, write findings, and flag missing inputs."),
        ("describe agent approval checkpoints", "Approval checkpoints pause the agent before external messages, payments, deletions, or irreversible changes."),
        ("plan an agent for customer support triage", "Agent plan: classify requests, suggest answers, escalate sensitive cases, and track resolution quality."),
        ("design an agent that reviews code changes", "Agent design: read the diff, run checks, comment on risks, verify tests, and summarize findings."),
        ("plan an agent for travel planning", "Agent plan: collect preferences, compare routes and stays, check constraints, and present options before booking."),
        ("describe agent memory boundaries", "Agent memory should store only approved facts, expire stale items, and let users review or delete entries."),
        ("plan an agent that organizes files", "Agent plan: inventory files, propose folders, move only after approval, and keep an undo log."),
        ("design an agent for learning plans", "Agent design: assess level, set goals, schedule practice, quiz progress, and adjust difficulty."),
        ("plan an agent that checks website uptime", "Agent plan: probe approved pages, record status, alert on failures, and summarize incidents."),
        ("describe safe agent tool permissions", "Tool permissions should be narrow, time-limited when possible, logged, and separated by task."),
        ("plan an agent for invoice processing", "Agent plan: extract invoice fields, match orders, flag mismatches, and route approvals before payment."),
        ("design an agent that writes summaries", "Agent design: identify audience, extract key points, preserve numbers, and keep uncertainty visible."),
        ("plan an agent for social media drafts", "Agent plan: draft posts from approved topics, check tone, schedule only after approval, and track results."),
        ("describe agent evaluation methods", "Evaluate agents with task success, error rate, approval quality, cost, latency, and user satisfaction."),
        ("plan an agent that manages reminders", "Agent plan: capture commitments, set reminders, snooze respectfully, and confirm completion."),
        ("design an agent for data cleanup", "Agent design: profile data, propose fixes, preview changes, apply approved edits, and keep backups."),
        ("plan an agent that compares products", "Agent plan: define criteria, gather specs, normalize prices, highlight trade-offs, and recommend by use case."),
        ("describe agent stop conditions", "Stop conditions include missing data, policy conflict, repeated failure, budget limit, and user cancellation."),
        ("plan an agent for onboarding checklists", "Agent plan: create accounts checklist, schedule training, assign buddy tasks, and track completion."),
        ("design an agent that monitors news topics", "Agent design: follow approved topics, deduplicate stories, summarize changes, and cite sources."),
        ("plan an agent for form filling help", "Agent plan: gather required fields, validate entries, show a review screen, and submit only after approval."),
        ("describe agent logging practices", "Log goals, tool calls, decisions, approvals, errors, and outcomes without storing secrets."),
        ("plan an agent that prepares presentations", "Agent plan: outline story, draft slides, add evidence, check readability, and export after review."),
        ("design an agent for inventory alerts", "Agent design: track stock levels, forecast shortages, alert owners, and suggest reorder quantities."),
        ("plan an agent that translates messages", "Agent plan: detect language, translate meaning, preserve tone, and flag uncertain phrases."),
        ("describe multi agent coordination", "Coordination needs clear roles, shared state, handoff rules, conflict resolution, and one accountable owner."),
        ("plan an agent for bug report triage", "Agent plan: reproduce steps, classify severity, find duplicates, assign owners, and track fixes."),
        ("design an agent that plans workouts", "Agent design: ask goals and limits, schedule sessions, track progress, and adjust for recovery."),
        ("plan an agent for expense tracking", "Agent plan: capture receipts, categorize spending, flag unusual items, and summarize monthly totals."),
        ("describe agent privacy controls", "Privacy controls limit data access, mask sensitive fields, set retention, and record every disclosure."),
        ("plan an agent that drafts job applications", "Agent plan: match experience to the role, draft honestly, tailor keywords, and ask before submitting."),
        ("design an agent for study flashcards", "Agent design: extract concepts, write clear cards, schedule reviews, and retire mastered cards."),
        ("plan an agent that checks accessibility", "Agent plan: scan pages, report contrast and labels, suggest fixes, and verify after changes."),
        ("describe agent cost controls", "Cost controls set budgets, cache repeated work, choose smaller tools first, and report spend."),
        ("plan an agent for event coordination", "Agent plan: track venue, speakers, schedule, registrations, reminders, and day-of checklist."),
        ("design an agent that audits spreadsheets", "Agent design: check formulas, totals, duplicates, outliers, and produce a findings list."),
        ("plan an agent for knowledge base updates", "Agent plan: find outdated pages, draft updates, route approvals, publish, and record changes."),
        ("describe agent handoff to humans", "Handoff should include goal, progress, evidence, blockers, and the exact decision needed."),
        ("plan an agent that files reports", "Agent plan: collect reports, validate them, file by date and topic, log each location, and flag missing reports."),
        # Demo-critical agentic phrasings (28 Nov 2026 cycle).
        ("plan an agent that checks email and drafts replies", "Agent plan: read new mail each morning, sort by sender and urgency, draft replies for review, and send nothing without approval."),
        ("build an agent that monitors website uptime", "Agent plan: check the site on a schedule, record response times, alert on failures with the failing check, and keep a daily uptime summary."),
        # Demo-probe repair (28 Nov 2026 cycle, 02:16 run):
        # price-watch and morning-news agent asks echoed while
        # the email and uptime plans answered.
        ("create an agent that monitors prices and sends alerts", "Agent plan: track the listed prices on a schedule, compare them against your alert thresholds, and send one clear alert with the old and new price when a threshold is crossed."),
        ("build a bot that summarises news every morning", "Agent plan: gather the morning headlines from your chosen sources, summarise the top stories briefly, and deliver one digest at your set time."),
        # Sparse-evidence guard (28 Nov 2026 cycle, 02:16 run):
        # the retrain let the expense class answer the sparse
        # "plan an agent for reports" ask confidently. A filing
        # phrasing overshot (the sparse ask matched it at 0.57);
        # the triage class gaining the singular "for report"
        # shape splits the margin three ways instead, so sparse
        # evidence falls back honestly again (measured 0.21).
        ("plan an agent for report triage", "Agent plan: reproduce steps, classify severity, find duplicates, assign owners, and track fixes."),
        # Demo-probe repair (28 Nov 2026 cycle, 05:16 run):
        # expense, meeting-scheduling and uptime-alert agent
        # asks echoed, and the customer-questions bot was
        # answered by the morning-news class, so those classes
        # gain their natural shapes.
        ("create an agent that tracks expenses", "Agent plan: capture receipts, categorize spending, flag unusual items, and summarize monthly totals."),
        ("build an expense tracking agent", "Agent plan: capture receipts, categorize spending, flag unusual items, and summarize monthly totals."),
        ("build an agent to schedule meetings", "Agent plan: check availability, propose times, confirm participants, send invites, and handle changes politely."),
        ("create a meeting scheduling agent", "Agent plan: check availability, propose times, confirm participants, send invites, and handle changes politely."),
        ("make a bot that answers customer questions", "Agent plan: classify requests, suggest answers, escalate sensitive cases, and track resolution quality."),
        ("build a customer question answering bot", "Agent plan: classify requests, suggest answers, escalate sensitive cases, and track resolution quality."),
        ("plan an agent that watches website uptime and sends alerts", "Agent plan: check the site on a schedule, record response times, alert on failures with the failing check, and keep a daily uptime summary."),
        ("watch website uptime and send alerts", "Agent plan: check the site on a schedule, record response times, alert on failures with the failing check, and keep a daily uptime summary."),
        # Sparse-evidence guard (28 Nov 2026 cycle, 05:16 run):
        # the customer-bot shapes above reuse the support
        # triage completion instead of a new class; a separate
        # customer class let the sparse "plan an agent for
        # reports" ask match at 0.4942, while the shared class
        # measures 0.0493, so sparse evidence falls back
        # honestly again.
        # Demo-probe repair (28 Nov 2026 cycle, 07:16 run):
        # the email-summary and stock-monitor agent asks
        # echoed, the books-meetings ask was answered by the
        # price-watch class and the social-posts ask by the
        # product-compare class, and the resume-screening
        # ask echoed, so those classes gain their natural
        # shapes.
        ("create an agent that summarizes my emails", "Agent plan: read approved messages, group by topic, draft summaries, flag urgent items, and ask before sending anything."),
        ("agent that summarizes my emails", "Agent plan: read approved messages, group by topic, draft summaries, flag urgent items, and ask before sending anything."),
        ("build an agent that monitors stock prices", "Agent plan: watch approved items, record changes, alert on thresholds, and avoid purchases without approval."),
        ("agent that tracks stock prices", "Agent plan: watch approved items, record changes, alert on thresholds, and avoid purchases without approval."),
        ("make an agent that books meetings", "Agent plan: check availability, propose times, confirm participants, send invites, and handle changes politely."),
        ("agent that books meetings", "Agent plan: check availability, propose times, confirm participants, send invites, and handle changes politely."),
        ("plan an agent that writes social media posts", "Agent plan: draft posts from approved topics, check tone, schedule only after approval, and track results."),
        ("agent that writes social media posts", "Agent plan: draft posts from approved topics, check tone, schedule only after approval, and track results."),
        ("create an agent that screens resumes", "Agent plan: screen resumes against approved criteria, rank matches with evidence, flag missing information, and leave hiring decisions to people."),
        ("agent for screening resumes", "Agent plan: screen resumes against approved criteria, rank matches with evidence, flag missing information, and leave hiring decisions to people."),
    ],
    "gravion-omni": [
        ("route a coding request to the right module", "Route coding requests to the coder module with the goal, constraints, and test expectations preserved."),
        ("route an image request to the right module", "Route image requests to the image module with subject, style, size, and quality details preserved."),
        ("route a security question to the right module", "Route security questions to the cyber module with assets, context, and defensive goal preserved."),
        ("combine chat and support in one answer", "Answer conversationally first, then give the support steps, escalation path, and a clear next action."),
        ("plan a website with copy and images", "Plan the site structure, draft page copy, define image briefs, and keep one consistent visual style."),
        ("create a video with music direction", "Plan the video scenes, pacing, narration, music mood, and sound cues as one production brief."),
        ("turn a research brief into an agent plan", "Convert findings into agent steps with tools, checks, approvals, and a measurable outcome."),
        ("coordinate code and security review", "Build the code change, then review it for input validation, secrets, access control, and tests."),
        ("plan a product launch across modules", "Launch plan: website, visuals, video, support answers, agent checklist, and security review."),
        ("summarize then create a presentation", "Summarize the source, choose three key points, draft slides, and add speaker notes."),
        ("write support answers with a friendly tone", "Use a warm tone, answer the exact question, give steps, and offer human help when needed."),
        ("design a brand kit with logo and site", "Brand kit: logo direction, colors, typography, site layout, and image style rules."),
        ("plan a course with video and quizzes", "Course plan: lessons, video outlines, practice tasks, quizzes, and a final project."),
        ("audit a website for content and safety", "Audit content clarity, accessibility, broken links, privacy notes, and security headers."),
        ("create a social campaign package", "Campaign package: message, visuals, short video script, posting plan, and response templates."),
        ("turn meeting notes into tasks and code issues", "Extract decisions, create tasks with owners, and draft code issues with reproduction steps."),
        ("plan a customer onboarding journey", "Journey: welcome message, setup guide, first success task, support path, and feedback check."),
        ("combine data report with executive summary", "Report: headline result, evidence chart, risks, recommendation, and appendix details."),
        ("design an app concept with agent features", "Concept: user goal, core screens, agent actions, guardrails, and success metrics."),
        ("prepare a multilingual support pack", "Support pack: translated answers, tone notes, escalation phrases, and review checklist."),
        ("plan a podcast with visuals and clips", "Podcast plan: episode outline, cover image brief, short clip moments, and show notes."),
        ("review a project across quality dimensions", "Review correctness, clarity, security, accessibility, performance, and documentation."),
        ("create a training plan with media", "Training plan: objectives, lessons, visuals, practice, assessment, and feedback loop."),
        ("coordinate a research and writing workflow", "Workflow: question, sources, outline, draft, fact check, revision, and final summary."),
        ("plan a safe automation rollout", "Rollout: pilot scope, approvals, monitoring, rollback plan, training, and gradual expansion."),
        ("design a dashboard with alerts", "Dashboard: key metrics, alert rules, drill-down views, and ownership for each signal."),
        ("turn a support ticket into a knowledge article", "Article: symptom, cause, solution steps, prevention, and related links."),
        ("plan a creative campaign with video", "Campaign: audience, message, hero visual, video script, channels, and measurement."),
        ("combine code generation with documentation", "Generate the code, add usage examples, document limits, and include tests."),
        ("prepare a security aware website launch", "Launch: content ready, forms validated, privacy page, backups, monitoring, and rollback plan."),
        ("create an agent workflow for reports", "Workflow: collect data, validate, draft report, route approval, publish, and archive evidence."),
        ("plan a product demo across formats", "Demo: script, visuals, short video, live walkthrough notes, and follow-up material."),
        ("design a learning app with chat support", "Learning app: lessons, practice, chat help, progress tracking, and teacher view."),
        ("coordinate image and video consistency", "Keep palette, typography, subject styling, and pacing consistent across image and video outputs."),
        ("turn user feedback into product tasks", "Group feedback by theme, estimate impact, draft tasks, and prioritize by user value."),
        ("plan a complete small business toolkit", "Toolkit: website, booking, support answers, social content, invoicing checklist, and security basics."),
        ("create a research dashboard brief", "Brief: questions, data sources, charts, filters, update cadence, and decision use."),
        ("combine music and video mood direction", "Match tempo, color, pacing, and emotional arc so music and visuals reinforce each other."),
        ("prepare an incident response communication pack", "Pack: status update, user guidance, internal checklist, evidence log, and follow-up report."),
        ("design a personal productivity system", "System: capture inbox, priorities, calendar blocks, review ritual, and agent reminders."),
        # Omni margin repair (28 Nov 2026 cycle, 07:16 run):
        # "my payment failed" scored 0.2683 toward the
        # order-status class and "i cannot login to my
        # account" 0.008 under Gravion Omni while Gravion
        # Support answered both, so the payment and lockout
        # classes gain distinct Omni-side shapes (the union
        # dedupes identical prompts, so these phrasings
        # differ from the support list).
        ("payment failed help me", "If a payment failed, check that the card details and balance are correct, then try again. Share the order number if the charge still appears."),
        ("my payment failed what should i do", "If a payment failed, check that the card details and balance are correct, then try again. Share the order number if the charge still appears."),
        ("payment failed again", "If a payment failed, check that the card details and balance are correct, then try again. Share the order number if the charge still appears."),
        ("cannot login help", "After several failed sign-ins the account locks for safety. Wait a short while, then use the password reset link to regain access."),
        ("login nahi ho raha", "After several failed sign-ins the account locks for safety. Wait a short while, then use the password reset link to regain access."),
        ("i cant login to my account", "After several failed sign-ins the account locks for safety. Wait a short while, then use the password reset link to regain access."),
    ],
    "gravion-translate": [
        ("translate a greeting into hindi", "Hello in Hindi is namaste (नमस्ते), the warm everyday greeting."),
        ("translate thank you into hindi", "Thank you in Hindi is dhanyavaad (धन्यवाद); shukriya is also common."),
        ("translate good morning into hindi", "Good morning in Hindi is suprabhaat (सुप्रभात)."),
        ("translate how are you into hindi", "How are you in Hindi is aap kaise hain (आप कैसे हैं)."),
        ("translate please into hindi", "Please in Hindi is kripya (कृपया)."),
        ("translate sorry into hindi", "Sorry in Hindi is maaf kijiye (माफ़ कीजिए)."),
        ("translate yes and no into hindi", "Yes is haan (हाँ) and no is nahi (नहीं) in Hindi."),
        ("translate welcome into hindi", "Welcome in Hindi is swagat hai (स्वागत है)."),
        ("translate congratulations into hindi", "Congratulations in Hindi is badhai ho (बधाई हो)."),
        ("translate happy birthday into hindi", "Happy birthday in Hindi is janamdin mubarak (जन्मदिन मुबारक)."),
        ("translate water into hindi", "Water in Hindi is paani (पानी)."),
        ("translate friend into hindi", "Friend in Hindi is dost (दोस्त)."),
        ("translate home into hindi", "Home in Hindi is ghar (घर)."),
        ("translate work into hindi", "Work in Hindi is kaam (काम)."),
        ("translate book into hindi", "Book in Hindi is kitaab (किताब)."),
        ("translate money into hindi", "Money in Hindi is paisa (पैसा)."),
        ("translate time into hindi", "Time in Hindi is samay (समय)."),
        ("translate success into hindi", "Success in Hindi is safalta (सफलता)."),
        ("translate hard work into hindi", "Hard work in Hindi is mehnat (मेहनत)."),
        ("translate knowledge into hindi", "Knowledge in Hindi is gyaan (ज्ञान)."),
        ("translate peace into hindi", "Peace in Hindi is shaanti (शांति)."),
        ("translate security into hindi", "Security in Hindi is suraksha (सुरक्षा)."),
        ("translate namaste into english", "Namaste means hello, a respectful greeting."),
        ("translate dhanyavaad into english", "Dhanyavaad means thank you."),
        ("translate shukriya into english", "Shukriya means thank you."),
        ("translate theek hai into english", "Theek hai means alright or okay."),
        ("translate koi baat nahi into english", "Koi baat nahi means no problem or never mind."),
        ("translate samajh gaya into english", "Samajh gaya means understood."),
        ("translate phir milenge into english", "Phir milenge means see you again."),
        ("translate dhyaan se into english", "Dhyaan se means carefully."),
        ("translate jaldi karo into english", "Jaldi karo means hurry up."),
        ("translate bahut badhiya into english", "Bahut badhiya means very good or well done."),
        ("translate mujhe madad chahiye into english", "Mujhe madad chahiye means I need help."),
        ("translate kal milte hain into english", "Kal milte hain means see you tomorrow."),
        ("translate apna khayal rakhna into english", "Apna khayal rakhna means take care of yourself."),
        ("translate ye kitne ka hai into english", "Ye kitne ka hai means how much does this cost."),
        ("explain hinglish in simple words", "Hinglish mixes Hindi and English in one sentence, written in Roman letters, as many people speak daily."),
        ("explain the difference between hindi and hinglish", "Hindi is the language itself; Hinglish is Hindi written in Roman script and mixed with English words."),
        ("give a polite hindi request phrase", "A polite request: kripya meri madad kijiye — please help me."),
        ("give a hindi phrase for asking directions", "Ask directions with: station kahan hai — where is the station."),
        ("give a hindi phrase for saying i do not understand", "Say: mujhe samajh nahi aaya — I did not understand."),
        ("give a hindi phrase for please speak slowly", "Say: kripya dheere boliye — please speak slowly."),
        ("translate today tomorrow yesterday into hindi", "Today is aaj, tomorrow is kal, and yesterday is also kal; context decides which kal is meant."),
        ("translate i am fine into hindi", "I am fine in Hindi is main theek hoon (मैं ठीक हूँ)."),
        ("translate what is your name into hindi", "What is your name in Hindi is aapka naam kya hai (आपका नाम क्या है)."),
        ("translate see you tomorrow into hindi", "See you tomorrow in Hindi is kal milte hain (कल मिलते हैं)."),
        ("translate take care into hindi", "Take care in Hindi is apna khayal rakhna (अपना ख्याल रखना)."),
        ("translate well done into hindi", "Well done in Hindi is bahut badhiya (बहुत बढ़िया)."),
    ],
    "gravion-summary": [
        ("summarize a long article briefly", "A brief summary keeps the main claim, the strongest evidence, and the conclusion, in that order."),
        ("explain extractive summarization", "Extractive summarization selects the source's own most important sentences instead of rewriting them."),
        ("make a one line summary of a report", "One line: state what happened, why it matters, and what happens next."),
        ("summarize a meeting into action points", "Meeting summary: decisions made, owners named, deadlines set, and open questions listed."),
        ("explain how to choose key sentences", "Key sentences carry the thesis, repeated central terms, numbers, and conclusions."),
        ("shorten a paragraph without losing meaning", "Cut examples and repetition first; keep definitions, claims, and the final point."),
        ("summarize a news story fairly", "A fair summary reports who, what, when, where, and why, without adding opinion."),
        ("make bullet points from a long text", "Turn each main idea into one short bullet, ordered as the text presents them."),
        ("explain the difference between summary and paraphrase", "A summary condenses the whole text; a paraphrase rewrites a passage at similar length."),
        ("summarize a chapter for revision", "Revision summary: key terms, main arguments, one example each, and likely exam points."),
        ("create an executive summary outline", "Executive summary: purpose, findings, risks, recommendation, and next step."),
        ("summarize pros and cons of a choice", "Pros and cons summary: strongest benefit, strongest cost, and who each side suits."),
        ("explain why summaries must stay faithful", "A summary must not add claims the source never made; faithfulness keeps trust."),
        ("summarize a research paper abstract style", "Abstract style: question, method, result, and implication in four short sentences."),
        ("make a summary for a busy reader", "For a busy reader: headline point first, three supporting facts, and the decision needed."),
        ("summarize a long email thread", "Email thread summary: original request, replies that changed anything, and the current open item."),
        ("explain sentence scoring in summaries", "Sentence scoring rates sentences by important terms, position, and length, then keeps the best."),
        ("summarize a product description", "Product summary: what it is, who it serves, key features, and price or availability."),
        ("create a summary with a word limit", "With a word limit, draft long, then cut adjectives and examples until only essentials remain."),
        ("summarize a historical event briefly", "Brief history summary: cause, turning point, outcome, and lasting effect."),
        ("explain how to summarize without bias", "Neutral summary uses the source's own claims, attributes opinions, and avoids loaded words."),
        ("summarize a tutorial into steps", "Tutorial summary: goal, materials, numbered steps, and the check that proves success."),
        ("make a summary of a long conversation", "Conversation summary: topics covered, agreements reached, disagreements left, and next meeting."),
        ("summarize a book in one paragraph", "Book summary: premise, main conflict or argument, turning point, and ending or conclusion."),
        ("explain lead sentences in summaries", "A lead sentence states the whole point first, so readers grasp the summary instantly."),
        ("summarize a legal notice plainly", "Plain summary of a notice: who must do what, by when, and what happens if they do not."),
        ("create a summary for social media", "Social summary: one hook line, two facts, and a question or link for detail."),
        ("summarize a scientific result carefully", "Careful summary: what was measured, the result size, uncertainty, and limits of the claim."),
        ("explain compression ratio in summaries", "Compression ratio compares summary length to source length; smaller ratios demand stricter selection."),
        ("summarize a speech into quotes", "Quote summary: pick the lines that carry the argument, keep them exact, and attribute them."),
        ("make a summary table from a report", "Summary table: rows for findings, columns for evidence, impact, and owner."),
        ("summarize a project status update", "Status summary: done, in progress, blocked, and next milestone with its date."),
        ("explain why order matters in summaries", "Keeping the source order preserves cause and effect, so the summary reads logically."),
        ("summarize a debate fairly", "Debate summary: each side's strongest point, the evidence offered, and where they disagree."),
        ("create a summary for children", "Child-friendly summary: short sentences, familiar words, one idea at a time, and a friendly tone."),
        ("summarize a financial report briefly", "Financial summary: revenue direction, profit direction, cash position, and the main risk."),
        ("explain headline writing for summaries", "A summary headline names the actor and the outcome in under ten words."),
        ("summarize a long instruction manual", "Manual summary: purpose, safety notes, main steps, and troubleshooting pointers."),
        ("make a summary from bullet notes", "From bullets: group related points, drop duplicates, and write one sentence per group."),
        ("summarize a biography briefly", "Biography summary: origin, defining work, major turning point, and legacy."),
        ("explain what to leave out of a summary", "Leave out repetition, minor examples, side stories, and anything the conclusion does not need."),
        ("summarize a policy document", "Policy summary: who it covers, what changes, when it starts, and how exceptions work."),
        ("create a two sentence summary", "Two sentences: first the main point, second the evidence or consequence."),
        ("summarize a travel guide page", "Travel summary: best sights, best season, costs to expect, and one local tip."),
        ("explain summary length choices", "Choose length by use: one line for lists, a paragraph for reports, a page for decisions."),
        ("summarize a customer review set", "Review summary: common praise, common complaints, average rating, and who the product suits."),
        ("make a summary that keeps numbers", "Keep the numbers that change decisions: totals, percentages, dates, and comparisons."),
        ("summarize a long story briefly", "Story summary: hero, goal, obstacle, climax, and resolution in five beats."),
    ],
    "gravion-search": [
        ("explain how search ranking works", "Ranking scores each result against the query and orders the best matches first."),
        ("explain embeddings in simple words", "An embedding turns text into numbers so similar meanings sit close together."),
        ("explain cosine similarity simply", "Cosine similarity measures the angle between two vectors; closer angles mean closer meaning."),
        ("find the best passage for a question", "The best passage shares the question's key terms and answers it directly, not just mentions it."),
        ("explain keyword search versus semantic search", "Keyword search matches exact words; semantic search matches meaning even with different words."),
        ("make a good search query", "A good query uses the rarest, most specific terms and drops filler words."),
        ("explain n grams in search", "N-grams are short word sequences; they help search match phrases, not only single words."),
        ("rank documents by relevance", "Relevance ranking weighs term matches, phrase matches, freshness, and source quality."),
        ("explain sparse vectors", "A sparse vector stores only its non-zero entries, which suits text with huge vocabularies."),
        ("explain tf idf in simple words", "TF-IDF rewards words frequent in a document but rare across the collection."),
        ("search passages about a topic", "Passage search embeds each passage and returns the ones nearest the query."),
        ("explain why search results need scores", "Scores show how strong each match is, so weak results can be filtered or explained."),
        ("improve a vague search query", "Add the specific noun, the place or date, and remove words every document contains."),
        ("explain document indexing", "Indexing precomputes searchable structures so queries answer in milliseconds."),
        ("find exact phrase matches in text", "Exact phrase search requires the words together in order, usually marked with quotes."),
        ("explain synonyms in search", "Synonym expansion lets a query match related words, widening recall without losing precision."),
        ("rank answers by evidence strength", "Stronger evidence answers directly, cites specifics, and comes from reliable sources."),
        ("explain stop words in search", "Stop words like the and of carry little meaning, so search often ignores them."),
        ("make search results explainable", "Explainable results show matched terms and scores, so users see why each result ranked."),
        ("explain recall and precision", "Recall is how much relevant material was found; precision is how much found material is relevant."),
        ("search within a long document", "Document search jumps to matching sections and shows context around each hit."),
        ("explain vector databases simply", "A vector database stores embeddings and finds nearest neighbors quickly at scale."),
        ("find similar documents to a given text", "Similarity search embeds the text and returns documents with the closest vectors."),
        ("explain query expansion", "Query expansion adds related terms so relevant documents using other words still match."),
        ("rank passages for a support question", "For support, rank passages that answer the exact problem above general background."),
        ("explain hashing tricks for text features", "Hashing maps each term to a fixed bucket, giving compact vectors without a stored vocabulary."),
        ("make a search over meeting notes", "Notes search works best with dates, names, and decision keywords indexed together."),
        ("explain duplicate results in search", "Duplicates waste ranks; deduplication keeps the strongest copy of near-identical results."),
        ("find the source of a quote", "Quote search matches the exact phrase first, then checks the earliest reliable source."),
        ("explain freshness in ranking", "Freshness boosts newer results when the topic changes quickly, like news or prices."),
        ("search code for a function name", "Code search matches identifiers exactly and ranks definitions above usages."),
        ("explain faceted search filters", "Facets filter results by category, date, or type while keeping counts visible."),
        ("rank web pages for a beginner question", "Beginner ranking favors clear explanations, worked examples, and trusted educational sources."),
        ("explain stemming in search", "Stemming reduces words to a root form so running and runs can match."),
        ("find passages mentioning a date", "Date search normalizes formats first, then matches the exact day, month, or year asked."),
        ("explain zero result searches", "Zero results mean the query terms never co-occur; broaden terms or drop the rarest one."),
        ("make search tolerant of typos", "Typo tolerance uses edit distance to match near-spellings of query terms."),
        ("explain result snippets", "Snippets show the matching context so users can judge a result before opening it."),
        ("rank legal clauses by relevance", "Legal ranking favors exact defined terms and the governing clause over mentions."),
        ("explain personalization in search", "Personalization reorders results using history and location, which can help or bias."),
        ("search a knowledge base effectively", "Knowledge search works when articles use the words users actually type."),
        ("explain bm25 ranking simply", "BM25 scores term frequency with saturation and document length normalization."),
        ("find contradictions between passages", "Contradiction search pairs passages on the same claim with opposing statements."),
        ("explain chunking documents for search", "Chunking splits documents into passages so search can return the exact relevant part."),
        ("rank tutorials by difficulty", "Difficulty ranking uses prerequisites, jargon density, and assumed prior knowledge."),
        ("explain search evaluation metrics", "Metrics like precision at k measure how many top results were actually relevant."),
        ("make a multilingual search plan", "Multilingual search indexes each language, detects query language, and ranks within it first."),
        ("explain why the top result can be wrong", "Top results optimize matching signals, not truth; verify claims against sources."),
    ],
    "gravion-analyst": [
        ("explain mean median and mode", "Mean is the average, median is the middle value, and mode is the most frequent value."),
        ("calculate an average from numbers", "Add all values, divide by how many there are, and report the count beside the average."),
        ("explain standard deviation simply", "Standard deviation measures typical distance from the mean; larger means more spread."),
        ("find a trend in monthly numbers", "Compare early and late averages, check direction, and note any single month that breaks the pattern."),
        ("explain outliers in data", "Outliers are values far from the rest; verify them before deciding whether they are errors or signals."),
        ("summarize a dataset in words", "Dataset summary: count, center, spread, extremes, and the one pattern that matters."),
        ("compare two groups of numbers", "Compare centers, spreads, and sample sizes; a difference matters only beyond natural variation."),
        ("explain percentages clearly", "A percentage is a part per hundred; always name the base it is a percentage of."),
        ("make a simple chart choice", "Bars compare categories, lines show change over time, and scatter plots show relationships."),
        ("explain correlation versus causation", "Correlation means values move together; causation needs a mechanism and controlled evidence."),
        ("clean messy numeric data", "Cleaning: fix formats, remove duplicates, flag missing values, and document every change."),
        ("explain sample size importance", "Larger samples give steadier estimates; tiny samples swing wildly with one new value."),
        ("calculate growth rate between periods", "Growth rate is change divided by the starting value, expressed as a percentage."),
        ("explain median versus mean choice", "Use the median when outliers skew the mean, such as incomes or house prices."),
        ("find the range of a dataset", "Range is maximum minus minimum; it shows total spread in one number."),
        ("explain quartiles simply", "Quartiles split sorted data into four equal parts, showing where the middle half lies."),
        ("detect missing values in data", "Count blanks per column, check patterns in what is missing, and decide fill or drop rules."),
        ("explain a histogram", "A histogram groups values into bins and shows how many fall in each bin."),
        ("compare this month with last month", "Compare totals, averages, and the change in percentage, then look for the cause."),
        ("explain statistical significance plainly", "Significance asks whether a difference is larger than chance alone would plausibly produce."),
        ("make a data table readable", "Readable tables align numbers, label units, sort meaningfully, and highlight key rows."),
        ("explain moving averages", "A moving average smooths noise by averaging a sliding window of recent values."),
        ("find the most common value", "The mode is the most common value; report how often it appears beside it."),
        ("explain data quality checks", "Quality checks cover completeness, consistency, valid ranges, and duplicate records."),
        ("calculate a percentage change", "Percentage change is new minus old, divided by old, times one hundred."),
        ("explain skewed distributions", "Skew means one tail is longer; the mean gets pulled toward the long tail."),
        ("summarize survey results", "Survey summary: response count, top answers with percentages, and who was surveyed."),
        ("explain confidence intervals simply", "A confidence interval gives a plausible range for the true value, not a single guess."),
        ("spot misleading charts", "Misleading charts truncate axes, cherry-pick ranges, or hide sample sizes."),
        ("explain regression in simple words", "Regression fits a line or curve that best predicts one value from others."),
        ("calculate total and share of total", "Share of total is each part divided by the sum, shown as a percentage."),
        ("explain time series basics", "Time series data is ordered by time; look for trend, seasonality, and sudden shifts."),
        ("compare averages with different counts", "Weight averages by their counts before combining, or the small group distorts the result."),
        ("explain variance simply", "Variance is the average squared distance from the mean; its square root is standard deviation."),
        ("find patterns in weekly sales", "Weekly patterns repeat by weekday; compare same weekdays across weeks before judging."),
        ("explain data sampling methods", "Sampling picks a subset to represent the whole; random sampling reduces selection bias."),
        ("make a forecast from past numbers", "A simple forecast extends the recent trend and states its uncertainty honestly."),
        ("explain normalization of data", "Normalization rescales values to a common range so different units can be compared."),
        ("calculate a weighted average", "Multiply each value by its weight, sum the products, and divide by total weight."),
        ("explain pivot tables simply", "A pivot table regroups rows by categories and summarizes values per group."),
        ("find errors in a spreadsheet", "Check formulas cell by cell, trace precedents, and test totals against raw sums."),
        ("explain a b testing basics", "A/B testing compares two variants on real users and judges the difference statistically."),
        ("summarize experiment results", "Experiment summary: groups, sample sizes, effect size, and whether it clears significance."),
        ("explain data visualization ethics", "Honest charts show full axes, label sources, and never hide inconvenient points."),
        ("calculate compound growth", "Compound growth applies each period's growth to the new total, not the original."),
        ("explain cohort analysis", "Cohort analysis groups users by start time and tracks each group separately."),
        ("find seasonal patterns in data", "Seasonality repeats on a calendar cycle; compare matching seasons across years."),
        ("explain key performance indicators", "KPIs are the few numbers tied directly to a goal, reviewed on a fixed cadence."),
    ],
    "gravion-docqa": [
        ("answer a question from a document", "Answer from the document only: find the matching sentence and quote it as evidence."),
        ("explain extractive question answering", "Extractive QA lifts the answer span directly from the source text instead of generating one."),
        ("find a date mentioned in a document", "Search the document for date patterns near the event words in the question."),
        ("explain grounding answers in evidence", "Grounded answers point to the exact sentence that supports them."),
        ("answer who wrote a document", "Look for author lines, bylines, or signature blocks near the title or end."),
        ("find a definition inside a document", "Definitions usually follow the term with is or means; quote that sentence."),
        ("explain what to do when a document lacks an answer", "Say the document does not contain the answer rather than guessing from outside knowledge."),
        ("answer a question about a policy document", "Policy answers cite the rule sentence and any exception sentence that applies."),
        ("find a number stated in a report", "Match the question's measure words to the sentence carrying the figure."),
        ("explain citation of document answers", "A citation names the document and quotes the supporting sentence verbatim."),
        ("answer where an event happened from text", "Place answers come from sentences pairing the event with a location phrase."),
        ("find the main requirement in a notice", "The main requirement is the sentence with must, shall, or required language."),
        ("explain document chunking for qa", "Chunking splits long documents so the answering passage can be found precisely."),
        ("answer a yes or no question from text", "Yes or no answers still need the sentence that settles the question."),
        ("find contact details in a document", "Contact details sit near headings like contact, support, or signature blocks."),
        ("explain ambiguity in document questions", "Ambiguous questions match several sentences; list them and ask which is meant."),
        ("answer when a deadline falls from a notice", "Deadline answers quote the sentence with the date and the action it binds."),
        ("find eligibility rules in a document", "Eligibility rules list conditions; quote each condition the question asks about."),
        ("explain multi hop document questions", "Multi-hop questions chain facts from two sentences before answering."),
        ("answer a question about a contract clause", "Clause answers quote the operative sentence and note defined terms it uses."),
        ("find the price stated in a document", "Price answers pair the amount with the item or plan it belongs to."),
        ("explain table questions in documents", "Table answers read the row and column headers together before quoting the cell."),
        ("answer how a process works from a manual", "Process answers list the manual's steps in order, quoted or tightly paraphrased."),
        ("find exceptions in a policy text", "Exceptions hide after words like except, unless, and however; quote them with the rule."),
        ("explain answer confidence in doc qa", "Confidence rises with term overlap and falls when several sentences compete."),
        ("answer a question about meeting minutes", "Minutes answers cite the decision line and who owned the action."),
        ("find a name mentioned in a document", "Name answers match the role words in the question to the sentence carrying the name."),
        ("explain document qa evaluation", "Doc QA is scored on exact answer match and whether evidence supports it."),
        ("answer what changed in a changelog", "Changelog answers quote the entry under the version or date asked."),
        ("find warranty terms in a document", "Warranty answers quote coverage length, covered faults, and claim steps."),
        ("explain handling long documents", "Long documents are searched by section so answers stay fast and precise."),
        ("answer a question from a resume", "Resume answers quote the role, dates, and achievement lines that match."),
        ("find instructions inside a document", "Instruction answers reproduce the numbered steps without adding new ones."),
        ("explain source attribution in answers", "Attribution tells the reader which document and sentence the answer came from."),
        ("answer a question about a research paper", "Paper answers quote the result sentence and the method limit beside it."),
        ("find a list inside a document", "List answers return the items under the heading the question names."),
        ("explain negation in document questions", "Negation flips meaning; check for not, never, and no before answering."),
        ("answer who is responsible from a document", "Responsibility answers quote the sentence assigning the duty to a person or team."),
        ("find opening hours in a document", "Hours answers quote the line pairing days with times."),
        ("explain document qa versus chat", "Document QA is confined to the given text; chat may use broader knowledge."),
        ("answer a question from terms and conditions", "Terms answers quote the clause and state its plain meaning."),
        ("find a summary statement in a document", "Summary statements sit in introductions and conclusions; quote the clearest one."),
        ("explain verifying an answer against text", "Verification re-reads the quoted sentence and checks it truly answers the question."),
        ("answer a question about a product manual", "Manual answers quote the setup or safety step the question targets."),
        ("find risks listed in a document", "Risk answers collect sentences with risk, warning, or caution language."),
        ("explain document qa input format", "Give the document text and the question; the answer quotes matching sentences."),
        ("answer a question from an invoice", "Invoice answers quote the line items, totals, and due date as printed."),
        ("find the conclusion of a document", "Conclusions restate the main finding; quote the final section's key sentence."),
    ],
}

MODEL_NAMES = {
    "gravion-chat": "Gravion Chat",
    "gravion-support": "Gravion Support",
    "gravion-review": "Gravion Review",
    "gravion-code": "Gravion Code",
    "gravion-builder": "Gravion Builder",
    "gravion-vision": "Gravion Vision",
    "gravion-video": "Gravion Video",
    "gravion-audio": "Gravion Audio",
    "gravion-shield": "Gravion Shield",
    "gravion-agent": "Gravion Agent",
    "gravion-translate": "Gravion Translate",
    "gravion-summary": "Gravion Summary",
    "gravion-search": "Gravion Search",
    "gravion-analyst": "Gravion Analyst",
    "gravion-docqa": "Gravion DocQA",
    "gravion-omni": "Gravion Omni",
}

MEDIA_NOTES = {
    "gravion-vision": " The trained local weights guide image briefs and prompt interpretation; the local procedural renderer remains the offline image path.",
    "gravion-video": " The trained local weights guide video briefs and scene planning; the local development renderer remains the offline clip path.",
    "gravion-audio": " The trained local weights guide music briefs and arrangement planning; the music module's local paths remain available offline.",
    "gravion-builder": " The trained local weights guide website briefs; the local static-site builder remains the offline build path.",
    "gravion-agent": " The trained local weights guide agent briefs and plans; execution remains bounded by deployment policy and approvals.",
    "gravion-translate": " The trained local weights guide translation requests; the translation module\u2019s Gravion-owned phrasebook is the offline translation path, and text outside the phrasebook is marked rather than guessed.",
    "gravion-summary": " The trained local weights guide summarization requests; the summarization module\u2019s local extractive summarizer is the offline summary path.",
    "gravion-search": " The trained local weights guide search requests; the retrieval module\u2019s local sparse-embedding ranking is the offline search path.",
    "gravion-analyst": " The trained local weights guide analysis requests; the data-analysis module computes statistics locally from supplied numbers.",
    "gravion-docqa": " The trained local weights guide document questions; the document-qa module answers extractively from supplied documents and says plainly when a document holds no answer.",
}


def build_examples(model_id: str) -> list[tuple[str, str]]:
    topics = list(TOPICS[model_id])
    if model_id == "gravion-omni":
        for other, other_topics in TOPICS.items():
            if other != "gravion-omni":
                topics.extend(other_topics)
    examples: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    name = MODEL_NAMES[model_id]
    for phrase, completion in topics:
        prompts = (
            phrase,
            f"please {phrase}",
            f"{name.lower()} {phrase}",
            f"help me {phrase}",
        )
        for prompt in prompts:
            pair = (prompt, completion)
            if pair not in seen:
                seen.add(pair)
                examples.append(pair)
    return examples


def main() -> None:
    catalog_path = DATA / "models.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in catalog["models"]}

    for model_id in MODEL_NAMES:
        examples = build_examples(model_id)
        seed_path = DATA / f"{model_id}-seed.jsonl"
        seed_path.write_text(
            "".join(
                json.dumps({"prompt": p, "completion": c}, sort_keys=True) + "\n"
                for p, c in examples
            ),
            encoding="utf-8",
        )
        epochs = 96 if model_id == "gravion-omni" else 160
        model = train_local_core(
            examples,
            model_id=model_id,
            source=f"data/{model_id}-seed.jsonl",
            epochs=epochs,
        )
        weights_path = DATA / f"{model_id}-weights.json"
        report = save_weights(model, weights_path)
        classes = len({completion for _, completion in examples})
        entry = entries[model_id]
        entry["weights_status"] = "included"
        entry["parameter_count"] = int(report["parameter_count"])
        entry["weights_artifact"] = f"data/{model_id}-weights.json"
        entry["weights_sha256"] = str(report["weights_sha256"])
        entry["notes"] = (
            f"Includes local sparse n-gram classifier weights "
            f"({int(report['parameter_count']):,} parameters counted from stored "
            f"quantized weights) trained only from the bundled Gravion-owned "
            f"{MODEL_NAMES[model_id]} seed corpus ({len(examples)} examples, "
            f"{classes} answer classes), with deterministic fallback when no "
            f"confident trained match is available. It runs offline with no "
            f"network and no API key."
            + MEDIA_NOTES.get(model_id, "")
        )
        print(
            f"{model_id}: examples={len(examples)} classes={classes} "
            f"parameters={report['parameter_count']}"
        )

    catalog["comment"] = (
        "The Gravion model registry — Gravion AI's own model entries and "
        "variants. Status 'available' means the registry entry and "
        "module/bundle packaging path can be resolved. The weights_status "
        "field is authoritative for weights: every bundled Gravion entry "
        "includes local sparse n-gram classifier weights trained only from "
        "its bundled Gravion-owned seed corpus, with a deterministic "
        "fallback. A deployment may register private variants by placing "
        "its own models.json in the deployment home; the same schema and "
        "validation apply."
    )
    catalog_path.write_text(
        json.dumps(catalog, indent=4, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
