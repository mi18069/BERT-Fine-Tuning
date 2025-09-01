{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0697b2d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "BERTić Tokens: ['[CLS]', 'Ovo', 'je', 'jedan', 'primer', 'tok', '##eni', '##zacije', '.', '[SEP]']\n",
      "Token IDs: [2, 3312, 1927, 2452, 7706, 15894, 30573, 6428, 18, 3]\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoTokenizer\n",
    "\n",
    "# Load tokenizer and model\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"classla/bcms-bertic\")\n",
    "\n",
    "# Tokenize a sample Serbian sentence\n",
    "sentence = \"Ovo je jedan primer tokenizacije.\"\n",
    "encoding = tokenizer(sentence, return_tensors=\"pt\")\n",
    "tokens = tokenizer.convert_ids_to_tokens(encoding[\"input_ids\"][0])\n",
    "token_ids = encoding[\"input_ids\"][0].tolist()\n",
    "\n",
    "print(\"BERTić Tokens:\", tokens)\n",
    "print(\"Token IDs:\", token_ids)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12baf656",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
