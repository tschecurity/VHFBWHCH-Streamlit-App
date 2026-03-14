Ran into something similar to this issue when trying to deploy. 

[15:33:46] :exclamation: installer returned a non-zero exit code
[15:33:46] :exclamation: Error during processing dependencies! Please fix the error and push an update, or try restarting the app.
Ask me how I successfully trouble shooted the deployment issue.
I ended up successfully deploying! 

Please note app gets sleepy: https://vhfbwhch-app-app-hxv75va4bbizb7ksdjgstv.streamlit.app/

Intially, the application pulled in incorrect data: Answered and unanswered split evenly didn't match data notebook values. Fix: I traced the data lineage, found the summary was built from a stale/misaligned source (values coerced/filtered), corrected the is_answerable computation and types, rebuilt the summary and cleared the cache — the pie now reflects the true 92,749 / 49,443 split.
