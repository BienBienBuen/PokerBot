import tarfile
import io
import os
import glob
import re
from pathlib import Path
from datetime import datetime

def parse_poker_hand(line):
    """
    Parse a poker hand from the compact format.
    Format appears to be: timestamp game_id n_players street1/amount1 street2/amount2 ... cards
    Returns a dictionary with hand information.
    """
    hand_info = {}
    
    try:
        parts = line.strip().split()
        if len(parts) < 7:  # Need at least timestamp, game_id, n_players, and some streets
            return None
            
        # Validate and convert numeric fields
        try:
            hand_info['timestamp'] = int(parts[0])
            hand_info['game_id'] = int(parts[1])
            hand_info['table_id'] = int(parts[2])
            hand_info['n_players'] = int(parts[3])
        except ValueError:
            return None  # Invalid numeric data
        
        # Process streets (betting rounds)
        streets = []
        cards = []
        for part in parts[4:]:
            if '/' in part:  # This is a street with betting
                try:
                    players, amount = part.split('/')
                    streets.append({
                        'active_players': int(players),
                        'pot_size': int(amount)
                    })
                except (ValueError, IndexError):
                    # Invalid street data, skip this street
                    continue
            else:  # This is a card
                # Validate card format (2-characters)
                if len(part) == 2:
                    cards.append(part)
        
        hand_info['streets'] = streets
        hand_info['cards'] = cards  # These appear to be community cards
        
        # Calculate some derived information
        if streets:
            hand_info['final_pot'] = streets[-1]['pot_size']
            hand_info['max_players_in_hand'] = max(street['active_players'] for street in streets)
            
        return hand_info
        
    except Exception as e:
        print(f"Error parsing line: {line}")
        print(f"Error details: {str(e)}")
        return None

def extract_poker_hands(data_dir):
    """
    Extract poker hands data from all .tgz files in the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing .tgz files
    """
    # Get all .tgz files
    tgz_files = glob.glob(os.path.join(data_dir, "*.tgz"))
    
    # Create output directory for data files
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a master file for all hands
    master_file_path = os.path.join(output_dir, 'all_poker_hands.csv')
    total_hands = 0
    
    with open(master_file_path, 'w') as master_file:
        # Write header
        master_file.write("timestamp,game_id,table_id,n_players,preflop_players,preflop_pot,flop_players,flop_pot,turn_players,turn_pot,river_players,river_pot,total_pot,cards\n")
        
        for tgz_file in tgz_files:
            print(f"Processing {tgz_file}...")
            
            try:
                # Extract game type from filename
                game_type = Path(tgz_file).stem.split('.')[0]
                
                with tarfile.open(tgz_file, 'r:gz') as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            f = tar.extractfile(member)
                            if f is not None:
                                content = f.read().decode('utf-8', errors='ignore')
                                
                                # Process each line (hand) in the file
                                for line in content.split('\n'):
                                    if not line.strip():
                                        continue
                                        
                                    # Parse poker hand data
                                    hand_info = parse_poker_hand(line)
                                    if hand_info:
                                        # Extract data for CSV
                                        timestamp = hand_info['timestamp']
                                        game_id = hand_info['game_id']
                                        table_id = hand_info['table_id']
                                        n_players = hand_info['n_players']
                                        
                                        # Initialize street data
                                        street_data = {
                                            'preflop': {'players': 0, 'pot': 0},
                                            'flop': {'players': 0, 'pot': 0},
                                            'turn': {'players': 0, 'pot': 0},
                                            'river': {'players': 0, 'pot': 0}
                                        }
                                        
                                        # Fill in street data
                                        streets = ['preflop', 'flop', 'turn', 'river']
                                        for i, street in enumerate(hand_info['streets'][:4]):  # Only first 4 streets
                                            street_data[streets[i]]['players'] = street['active_players']
                                            street_data[streets[i]]['pot'] = street['pot_size']
                                        
                                        # Format cards
                                        cards = ' '.join(hand_info['cards'])
                                        
                                        # Write CSV line, ensuring all fields have valid values
                                        final_pot = hand_info.get('final_pot', 0)  # Default to 0 if missing
                                        cards = cards if cards else ""  # Empty string if no cards
                                        csv_line = f"{timestamp},{game_id},{table_id},{n_players},"
                                        csv_line += f"{street_data['preflop']['players']},{street_data['preflop']['pot']},"
                                        csv_line += f"{street_data['flop']['players']},{street_data['flop']['pot']},"
                                        csv_line += f"{street_data['turn']['players']},{street_data['turn']['pot']},"
                                        csv_line += f"{street_data['river']['players']},{street_data['river']['pot']},"
                                        csv_line += f"{final_pot},{cards}\n"
                                        
                                        master_file.write(csv_line)
                                        total_hands += 1
                                        
                                        if total_hands % 10000 == 0:
                                            print(f"Processed {total_hands} hands...")
                                    
                print(f"Processed {tgz_file}")
            except Exception as e:
                print(f"Error processing {tgz_file}: {str(e)}")
                
    print(f"\nTotal hands processed: {total_hands}")
    print(f"Data saved to: {master_file_path}")
    print("The data is saved in CSV format with the following columns:")
    print("  timestamp, game_id, table_id, n_players, preflop_players, preflop_pot,")
    print("  flop_players, flop_pot, turn_players, turn_pot, river_players, river_pot,")
    print("  total_pot, cards")

def debug_single_file(file_path):
    """
    Debug a single poker hand file by printing its contents.
    """
    print(f"Debugging file: {file_path}")
    hands = []
    
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        content = f.read().decode('utf-8', errors='ignore')
                        
                        # Process each line (hand) in the file
                        for line in content.split('\n'):
                            if not line.strip():
                                continue
                            
                            # Store original line and parsed data
                            hand_info = parse_poker_hand(line)
                            if hand_info:
                                hands.append({
                                    'original': line.strip(),
                                    'parsed': hand_info
                                })
        
        # Print results
        print(f"\nFound {len(hands)} hands. First 10 hands:")
        for i, hand in enumerate(hands[:10]):
            print(f"\nHand #{i+1}")
            print(f"Original: {hand['original']}")
            print("Parsed:")
            for key, value in hand['parsed'].items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    print("Starting poker hand extraction...")
    print("-" * 50)
    
    # First debug the specific file
    debug_file = "/Users/bx/Downloads/IRCdata/holdem.200012.tgz"
    print("\nDEBUGGING SINGLE FILE")
    print("=" * 50)
    debug_single_file(debug_file)
    
    # Then process all files
    print("\nPROCESSING ALL FILES")
    print("=" * 50)
    irc_data_path = "/Users/bx/Downloads/IRCdata"
    extract_poker_hands(irc_data_path)