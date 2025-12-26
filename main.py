"""
Command Line Interface for the Data Pipeline.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.pipeline import DataPipeline

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🧠 NeuroPipe - Autonomous Document-to-LLM Data Factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF file
  python main.py process document.pdf --name my_dataset
  
  # Process all PDFs in a directory
  python main.py batch ./documents --name combined_dataset
  
  # Check pipeline health
  python main.py health
  
  # Start the web interface
  python main.py serve --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single PDF file')
    process_parser.add_argument('file', type=str, help='Path to PDF file')
    process_parser.add_argument('--name', '-n', type=str, help='Dataset name', default=None)
    process_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process all PDFs in a directory')
    batch_parser.add_argument('directory', type=str, help='Directory containing PDFs')
    batch_parser.add_argument('--name', '-n', type=str, help='Dataset name', default='dataset')
    batch_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check pipeline health')
    health_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web interface')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', '-p', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')

    # Distill command
    distill_parser = subparsers.add_parser('distill', help='Distill generated data for training')
    distill_parser.add_argument('--name', '-n', type=str, help='Batch name to process', default='combined')
    distill_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')

    # Train command
    train_parser = subparsers.add_parser('train', help='Fine-tune Sarvam-1 model')
    train_parser.add_argument('--data', '-d', type=str, help='Path to training data (jsonl)', default=None)
    train_parser.add_argument('--config', '-c', type=str, help='Config file path', default='config.yaml')
    
    args = parser.parse_args()
    
    if not args.command:
        console.print(Panel.fit(
            "[bold cyan]🧠 NeuroPipe Data Engine[/bold cyan]\n\n"
            "Autonomous pipeline to convert docs into high-quality LLM training data\n"
            "Features: [bold]Vision OCR • Synthetic Labeling • Data Distillation[/bold]\n\n"
            "[dim]Run 'python main.py --help' for usage information.[/dim]",
            title="Welcome"
        ))
        parser.print_help()
        return
    
    if args.command == 'health':
        pipeline = DataPipeline(args.config)
        healthy = pipeline.check_health()
        sys.exit(0 if healthy else 1)
    
    elif args.command == 'process':
        if not Path(args.file).exists():
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            sys.exit(1)
            
        pipeline = DataPipeline(args.config)
        
        if not pipeline.check_health():
            console.print("[red]Pipeline health check failed. Please ensure Ollama is running.[/red]")
            sys.exit(1)
        
        result = pipeline.process_file(args.file, args.name)
        
        if result.success:
            console.print(f"\n[bold green]✅ Success![/bold green] Processed in {result.duration_seconds:.1f}s")
            console.print(f"   Total samples generated: {result.total_samples}")
        else:
            console.print(f"\n[bold red]❌ Failed[/bold red]")
            for error in result.errors:
                console.print(f"   • {error}")
            sys.exit(1)
    
    elif args.command == 'batch':
        if not Path(args.directory).is_dir():
            console.print(f"[red]Error: Directory not found: {args.directory}[/red]")
            sys.exit(1)
            
        pipeline = DataPipeline(args.config)
        
        if not pipeline.check_health():
            console.print("[red]Pipeline health check failed. Please ensure Ollama is running.[/red]")
            sys.exit(1)
        
        result = pipeline.process_directory(args.directory, args.name)
        
        if result.success:
            console.print(f"\n[bold green]✅ Success![/bold green] Processed {result.documents_processed} documents in {result.duration_seconds:.1f}s")
            console.print(f"   Total samples generated: {result.total_samples}")
        else:
            console.print(f"\n[bold red]❌ Failed[/bold red]")
            for error in result.errors:
                console.print(f"   • {error}")
            sys.exit(1)
    
    elif args.command == 'serve':
        console.print(f"[bold cyan]Starting web interface on http://{args.host}:{args.port}[/bold cyan]")
        import uvicorn
        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=True)

    elif args.command == 'distill':
        from src.distiller import DataDistiller
        
        distiller = DataDistiller(DataPipeline(args.config).config) 
        distiller.distill_batch(args.name)

    elif args.command == 'train':
        from src.train import SarvamTrainer
        
        # Check for data path, default to distilled path if not provided
        data_path = args.data
        if not data_path:
            # Try to infer based on name or defaults
            data_path = "datasets/distilled_train.jsonl"
            
        trainer = SarvamTrainer(DataPipeline(args.config).config)
        trainer.train(data_path)


if __name__ == "__main__":
    main()
