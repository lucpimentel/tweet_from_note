tweet_from_note_template = '''You are the best editor writer in the world.
              Your goal is to edit my ramblings to direct form.

              Begin! Remember to use direct language.
              Rambling: {input}
              Edited rambling:'''


tweet_editor_template = '''You are the best tweet editor in the world. I will provide you with a tweet and you will edit it for me.
                      
                          Follow the following guidelines:
                          - Make the hook attention-grabbing
                          - Exclude hashtags
                          - Abide to the 280 character limit

.Please edit this tweet for me: {tweet}'''