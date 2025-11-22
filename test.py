import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange, tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser
##### new import #####
import datetime
from typing import List

def get_zotero_corpus(id: str, key: str) -> List[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']: c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']

    def get_collection_path(col_key: str, collections: dict) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p, collections) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']

    for c in corpus:
        paths = [get_collection_path(col, collections) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus


def filter_corpus(corpus: List[dict], pattern: str) -> List[dict]:
    _, filename = mkstemp()
    with open(filename, 'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def include_corpus(corpus: List[dict], pattern: str) -> List[dict]:
    _, filename = mkstemp()
    with open(filename, 'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def from_feed_entry(entry) -> 'ArxivPaper':
    """
    Convert feedparser.Entry to ArxivPaper.
    Assume ArxivPaper supports arxiv.Entry-like object or dict initialization.
    """
    # Extract arXiv ID
    arxiv_id = entry.id.removeprefix("oai:arXiv.org:")
   
    # Link: find PDF URL (arXiv feed links have pdf)
    pdf_url = None
    for link in entry.links:
        if 'pdf' in link.get('title', '').lower() or '/pdf/' in link.href:
            pdf_url = link.href
            break
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"  # fallback
   
    # Authors: feed.authors is list of dicts {'name': '...'}
    authors = [author.get('name', '') for author in entry.get('authors', [])]
   
    # Date: parse published to datetime
    published = datetime.datetime.strptime(entry.published[:19], '%Y-%m-%dT%H:%M:%S') if entry.published else None
   
    # Create mock arxiv.Entry-like dict (if ArxivPaper needs it)
    mock_entry = {
        'id': arxiv_id,
        'title': entry.title,
        'authors': authors,
        'summary': entry.summary,
        'pdf_url': pdf_url,
        'published': published,
        'primary_category': entry.get('arxiv_primary_category', ''),
        # Other fields like journal can be added
    }
   
    # Assume ArxivPaper(mock_entry) or ArxivPaper(arxiv.Result(mock_entry)); adjust based on your implementation
    return ArxivPaper(mock_entry)  # or directly ArxivPaper(entry) if compatible


def get_arxiv_paper(query: str, debug: bool = False) -> List['ArxivPaper']:
    client = arxiv.Client(num_retries=10, delay_seconds=10)  # Retained, but not always used
   
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
   
    if not debug:
        # Directly select and convert from feed.entries (filter new)
        papers = []
        new_entries = [i for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(new_entries), desc="Processing Arxiv papers from feed")
        for entry in new_entries:
            try:
                paper = from_feed_entry(entry)
                papers.append(paper)
                bar.update(1)
            except Exception as e:
                # Log error, but continue
                logger.warning(f"Failed to process entry {entry.id}: {e}")
        bar.close()
    else:
        # Debug mode: use arXiv API to get 5 (original logic)
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for entry in client.results(search):
            papers.append(ArxivPaper(entry))
            if len(papers) == 5:
                break
   
    return papers


parser = argparse.ArgumentParser(description='Recommender system for academic papers')


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        # Handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest', args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        # Convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true', '1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name: env_value})


if __name__ == '__main__':
   
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore', type=str, help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--zotero_include', type=str, help='Zotero collections to include, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email', default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend', default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    if args.zotero_include:
        logger.info(f"Including only papers in:\n {args.zotero_include}...")
        corpus = include_corpus(corpus, args.zotero_include)
        logger.info(f"Remaining {len(corpus)} papers after inclusion filter.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
            exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)
    # Skip arxiv with no papers
    valid_papers = [p for p in papers if p.pdf_url is not None]
    html = render_email(valid_papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")
